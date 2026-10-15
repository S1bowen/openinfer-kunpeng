#![cfg(feature = "comm")]
use std::net::TcpListener;
use std::time::Duration;

use pegainfer_kutacc::comm::bootstrap::Bootstrap;
use pegainfer_kutacc::comm::bootstrap::oob_allgather;

#[test]
fn rank_order_in_place_and_transpose() {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    drop(l);
    let workers: Vec<_> = (0..4)
        .map(|rank| {
            std::thread::spawn(move || {
                let mut b =
                    Bootstrap::connect(addr, rank, 4, 734523, Duration::from_secs(5)).unwrap();
                for step in 0..20 {
                    let mut recv = [0u64; 4];
                    recv[rank] = (100 * step + rank) as u64;
                    let code = unsafe {
                        oob_allgather(
                            std::ptr::null(),
                            recv.as_mut_ptr().cast(),
                            1,
                            b.context(),
                            2,
                        )
                    };
                    assert_eq!(code, 0);
                    assert_eq!(recv, std::array::from_fn(|r| (100 * step + r) as u64));
                    let send: Vec<u8> = (0..4).map(|r| (rank * 4 + r) as u8).collect();
                    let mut recv = [0; 4];
                    b.alltoall(&send, &mut recv).unwrap();
                    assert_eq!(recv, std::array::from_fn(|r| (r * 4 + rank) as u8));
                    b.barrier().unwrap();
                }
            })
        })
        .collect();
    for w in workers {
        w.join().unwrap();
    }
}

#[test]
fn zero_byte_barrier_waits_for_peer() {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    drop(l);
    let (tx, rx) = std::sync::mpsc::channel();
    let root = std::thread::spawn(move || {
        let mut b = Bootstrap::connect(addr, 0, 2, 4321, Duration::from_secs(5)).unwrap();
        rx.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        b.barrier().unwrap();
    });
    let mut b = Bootstrap::connect(addr, 1, 2, 4321, Duration::from_secs(5)).unwrap();
    let now = std::time::Instant::now();
    tx.send(()).unwrap();
    b.barrier().unwrap();
    assert!(now.elapsed() >= Duration::from_millis(90));
    root.join().unwrap();
}

#[test]
fn mismatched_collective_poisoned() {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    drop(l);
    let root = std::thread::spawn(move || {
        let mut b = Bootstrap::connect(addr, 0, 2, 98, Duration::from_millis(500)).unwrap();
        assert!(b.allgather(&[1], &mut [0; 2]).is_err());
        assert!(b.barrier().unwrap_err().to_string().contains("poisoned"));
    });
    let mut b = Bootstrap::connect(addr, 1, 2, 98, Duration::from_millis(500)).unwrap();
    assert!(b.barrier().is_err());
    root.join().unwrap();
}
