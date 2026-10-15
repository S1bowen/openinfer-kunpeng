//! Real KuPL shared-memory windows and KuTACC alltoall, with Rust TCP bootstrap.
//! Run on allocated 920F hardware; launches eight independent rank processes.
#[cfg(all(feature = "comm", target_arch = "aarch64", target_os = "linux"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::ffi::c_void;
    use std::net::TcpListener;
    use std::process::Command;
    use std::time::Duration;

    use pegainfer_kutacc::comm::bootstrap;
    use pegainfer_kutacc::comm::kupl;
    use pegainfer_kutacc::comm::raw;
    const WORLD: usize = 8;
    const H: usize = 4;
    const W: usize = WORLD * 32;
    let args: Vec<_> = std::env::args().collect();
    if args.len() == 1 {
        // Reserve an endpoint before starting rank zero. The cookie prevents an
        // unrelated job from joining if another process races to bind this port.
        let l = TcpListener::bind("127.0.0.1:0")?;
        let address = l.local_addr()?;
        drop(l);
        let cookie = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos() as u64;
        let mut children = Vec::new();
        for rank in 0..WORLD {
            children.push(
                Command::new(std::env::current_exe()?)
                    .env("KUPL_EXECUTOR_BACKEND", "pthread")
                    .args([rank.to_string(), address.to_string(), cookie.to_string()])
                    .spawn()?,
            );
        }
        let deadline = std::time::Instant::now() + Duration::from_secs(90);
        let mut failed = false;
        loop {
            let mut complete = true;
            for child in &mut children {
                match child.try_wait()? {
                    Some(s) => failed |= !s.success(),
                    None => complete = false,
                }
            }
            if complete {
                break;
            }
            if failed || std::time::Instant::now() > deadline {
                for c in &mut children {
                    let _ = c.kill();
                    let _ = c.wait();
                }
                return Err("rank failed or native collective timed out".into());
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        if failed {
            return Err("rank failed".into());
        }
        println!(
            "PASS: 8 Rust ranks, KuPL windows + KuTACC alltoall/allreduce/allgather/reduce-scatter"
        );
        return Ok(());
    }
    pegainfer_kutacc::comm::check_hardware()?;
    let rank: usize = args[1].parse()?;
    let mut boot = bootstrap::Bootstrap::connect(
        args[2].parse()?,
        rank,
        WORLD,
        args[3].parse()?,
        Duration::from_secs(20),
    )?;
    let mut callbacks = kupl::ShmCallbacks {
        allgather: bootstrap::oob_allgather,
        barrier: bootstrap::oob_barrier,
    };
    let mut comm = std::ptr::null_mut();
    let mut win = std::ptr::null_mut();
    let mut base: *mut c_void = std::ptr::null_mut();
    fn check(code: i32) {
        assert_eq!(code, 0, "native communication failed");
    }
    // Every pointer below is backed by the collective window; its lifecycle is
    // kept entirely on this thread. Final barrier precedes local teardown.
    unsafe {
        use std::sync::atomic::AtomicUsize;
        use std::sync::atomic::Ordering;
        unsafe extern "C" fn task(arg: *mut c_void) {
            unsafe { &*arg.cast::<AtomicUsize>() }.store(42, Ordering::Release);
        }
        let result = AtomicUsize::new(0);
        if rank == 0 {
            eprintln!("KuPL executors={}", kupl::kupl_get_num_executors());
        }
        let mut executor = 1;
        let group = kupl::kupl_egroup_create(&mut executor, 1);
        if rank == 0 {
            eprintln!("egroup created");
        }
        let queue = kupl::kupl_queue_create();
        if rank == 0 {
            eprintln!("queue created");
        }
        assert!(!group.is_null() && !queue.is_null());
        let mut item = kupl::QueueItem {
            field_mask: 2,
            func: task,
            args: (&result as *const AtomicUsize).cast_mut().cast(),
            name: std::ptr::null(),
            egroup: group,
            args_size: 0,
        };
        check(kupl::kupl_queue_submit(queue, &mut item));
        if rank == 0 {
            eprintln!("queue submitted");
        }
        check(kupl::kupl_queue_wait(queue));
        if rank == 0 {
            eprintln!("queue waited");
        }
        assert_eq!(result.load(Ordering::Acquire), 42);
        kupl::kupl_queue_destroy(queue);
        kupl::kupl_egroup_destroy(group);
        if rank == 0 {
            eprintln!("queue PASS");
        }
        check(kupl::kupl_shm_comm_create(
            WORLD as i32,
            rank as i32,
            std::process::id() as i32,
            &mut callbacks,
            boot.context(),
            &mut comm,
        ));
        check(kupl::kupl_shm_win_alloc(65536, comm, &mut base, &mut win));
        let mut peers = vec![std::ptr::null_mut(); WORLD];
        for (r, p) in peers.iter_mut().enumerate() {
            check(kupl::kupl_shm_win_query(win, r as i32, p));
        }
        let mut request = std::ptr::null_mut();
        let mut scratch = 0;
        check(raw::pegainfer_comm_shm_alltoall_request_create(
            rank as i32,
            WORLD as i32,
            H * W,
            0,
            &mut scratch,
            &mut request,
        ));
        assert_eq!(scratch, 0);
        check(raw::pegainfer_comm_shm_alltoall_request_init(
            peers.as_mut_ptr(),
            win,
            win,
            win,
            win,
            request,
        ));
        for round in 0..20 {
            let input: Vec<u8> = (0..H * W)
                .map(|i| ((rank * 13 + i + round * 7) % 251) as u8)
                .collect();
            std::ptr::copy_nonoverlapping(input.as_ptr(), base.cast(), input.len());
            let out = base.cast::<u8>().add(H * W);
            check(raw::pegainfer_comm_shm_alltoall2D(
                peers.as_mut_ptr(),
                out.cast(),
                H as i64,
                W as i64,
                0,
                true,
                request,
            ));
            for src in 0..WORLD {
                for row in 0..H {
                    for c in 0..W / WORLD {
                        let got = *out.add((src * H + row) * (W / WORLD) + c);
                        let expected =
                            ((src * 13 + row * W + rank * (W / WORLD) + c + round * 7) % 251) as u8;
                        assert_eq!(got, expected, "rank={rank} src={src} round={round}");
                    }
                }
            }
            check(kupl::kupl_shm_fence(win));
        }
        check(raw::pegainfer_comm_shm_alltoall_request_destroy(request));
        // All groups share the node fence in this smoke test. Production callers
        // supply separate die/socket windows matching their rank topology.
        if rank == 0 {
            eprintln!("alltoall PASS");
        }
        const N: usize = 1024;
        let mut extra: Vec<*mut c_void> = peers
            .iter()
            .map(|p| p.cast::<u8>().add(32768).cast())
            .collect();
        let mut reduce = std::ptr::null_mut();
        check(raw::pegainfer_comm_shm_allreduce_request_create(
            rank as i32,
            WORLD as i32,
            N,
            2,
            &mut scratch,
            &mut reduce,
        ));
        assert!(scratch <= 32768);
        check(raw::pegainfer_comm_shm_allreduce_request_init(
            extra.as_mut_ptr(),
            win,
            win,
            win,
            win,
            reduce,
        ));
        for round in 0..4 {
            let value = pegainfer_kutacc::Bf16::from_f32((rank + round) as f32);
            for i in 0..N {
                base.cast::<pegainfer_kutacc::Bf16>().add(i).write(value);
            }
            check(kupl::kupl_shm_fence(win));
            check(raw::pegainfer_comm_shm_allreduce(
                peers.as_mut_ptr(),
                N,
                reduce,
            ));
            for i in 0..N {
                assert_eq!(
                    (*base.cast::<pegainfer_kutacc::Bf16>().add(i)).to_f32(),
                    (28 + 8 * round) as f32
                );
            }
            check(kupl::kupl_shm_fence(win));
        }
        check(raw::pegainfer_comm_shm_allreduce_request_destroy(reduce));
        if rank == 0 {
            eprintln!("allreduce PASS");
        }
        let mut scatter = std::ptr::null_mut();
        check(raw::pegainfer_comm_shm_reduce_scatter_request_create(
            rank as i32,
            WORLD as i32,
            2,
            &mut scratch,
            &mut scatter,
        ));
        assert!(scratch <= 32768);
        check(raw::pegainfer_comm_shm_reduce_scatter_request_init(
            extra.as_mut_ptr(),
            win,
            win,
            win,
            scatter,
        ));
        for i in 0..N {
            base.cast::<pegainfer_kutacc::Bf16>()
                .add(i)
                .write(pegainfer_kutacc::Bf16::from_f32(rank as f32));
        }
        check(kupl::kupl_shm_fence(win));
        check(raw::pegainfer_comm_shm_reduce_scatter(
            peers.as_mut_ptr(),
            8,
            128,
            scatter,
        ));
        for i in rank * 128..(rank + 1) * 128 {
            assert_eq!(
                (*base.cast::<pegainfer_kutacc::Bf16>().add(i)).to_f32(),
                28.0
            );
        }
        check(kupl::kupl_shm_fence(win));
        check(raw::pegainfer_comm_shm_reduce_scatter_request_destroy(
            scatter,
        ));
        if rank == 0 {
            eprintln!("reduce-scatter PASS");
        }
        let mut gather = std::ptr::null_mut();
        check(raw::pegainfer_comm_shm_allgather_request_create(
            rank as i32,
            WORLD as i32,
            &mut gather,
        ));
        check(raw::pegainfer_comm_shm_allgather_request_init(
            win, win, win, gather,
        ));
        let mut remote_send: Vec<*mut u8> = peers.iter().map(|p| p.cast()).collect();
        let mut remote_recv: Vec<*mut u8> =
            peers.iter().map(|p| p.cast::<u8>().add(8192)).collect();
        for i in 0..64 {
            base.cast::<u8>().add(i).write((rank * 10 + i) as u8);
        }
        check(kupl::kupl_shm_fence(win));
        check(raw::pegainfer_comm_shm_batch_allgather(
            1,
            base,
            64,
            base.cast::<u8>().add(8192).cast(),
            512,
            1,
            remote_send.as_mut_ptr(),
            remote_recv.as_mut_ptr(),
            8192,
            gather,
            false,
            false,
        ));
        for r in 0..WORLD {
            for i in 0..64 {
                assert_eq!(
                    *base.cast::<u8>().add(8192 + r * 64 + i),
                    (r * 10 + i) as u8
                );
            }
        }
        check(kupl::kupl_shm_fence(win));
        check(raw::pegainfer_comm_shm_allgather_request_destroy(gather));
        if rank == 0 {
            eprintln!("allgather PASS");
        }
        boot.barrier()?;
        check(kupl::kupl_shm_win_free(win));
        check(kupl::kupl_shm_comm_destroy(comm));
        if std::env::var_os("PEGAINFER_TEST_RDMA").is_some() {
            let mut rdma = std::ptr::null_mut();
            let mut cb = bootstrap::RDMA_CALLBACKS;
            check(raw::pegainfer_comm_kurmcl_comm_create(
                WORLD as i32,
                rank as i32,
                (&mut cb as *mut bootstrap::RdmaCallbacks).cast(),
                boot.context(),
                &mut rdma,
            ));
            let mut send = vec![0u64; 8];
            let mut recv = vec![0u64; 8 * WORLD];
            check(raw::pegainfer_comm_kurmcl_allgather_full_init(
                send.as_mut_ptr().cast(),
                64,
                recv.as_mut_ptr().cast(),
                64,
                rdma,
            ));
            for step in 0..5 {
                send.fill((step * WORLD + rank) as u64);
                check(raw::pegainfer_comm_kurmcl_allgather_full(
                    send.as_mut_ptr().cast(),
                    64,
                    recv.as_mut_ptr().cast(),
                    64,
                    rdma,
                ));
                for r in 0..WORLD {
                    assert_eq!(&recv[r * 8..(r + 1) * 8], &[(step * WORLD + r) as u64; 8]);
                }
                boot.barrier()?;
            }
            check(raw::pegainfer_comm_kurmcl_allgather_full_finalize());
            // Upstream KURMCL has process-global ownership and no destroy API.
            // Rank subprocess exit is its cleanup boundary, as in SGLang.
            boot.barrier()?;
            println!("rank {rank}: RDMA allgather PASS");
        }
    }
    Ok(())
}
#[cfg(not(all(feature = "comm", target_arch = "aarch64", target_os = "linux")))]
fn main() {
    panic!("comm_smoke requires --features comm on Linux aarch64 hardware");
}
