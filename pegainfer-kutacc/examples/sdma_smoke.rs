//! Real SDMA transfer on an exclusively allocated SDK channel (no shared IDs).
#[cfg(all(feature = "sdma", target_arch = "aarch64", target_os = "linux"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::alloc::Layout;
    use std::alloc::alloc_zeroed;
    use std::alloc::dealloc;
    use std::fs::OpenOptions;
    use std::os::fd::AsRawFd;

    use pegainfer_kutacc::comm::sdma;
    pegainfer_kutacc::comm::check_hardware()?;
    // On an uncertain SDK error, exit the subprocess without freeing registered
    // memory or a possibly active channel. No Rust allocation owns these buffers.
    unsafe {
        assert_eq!(
            sdma::pegainfer_sdma_task_size(),
            std::mem::size_of::<sdma::Task>()
        );
        let device = sdma::sdma_nearest_id();
        if device < 0 {
            return Err(format!("sdma_nearest_id={device}").into());
        }
        let f = OpenOptions::new()
            .read(true)
            .write(true)
            .open(format!("/dev/sdma{device}"))?;
        let fd = f.as_raw_fd();
        let mut pid = 0;
        fn check(code: i32) -> Result<(), Box<dyn std::error::Error>> {
            if code == 0 {
                Ok(())
            } else {
                Err(format!("SDMA status {code}").into())
            }
        }
        check(sdma::sdma_get_process_id(fd, &mut pid))?;
        let layout = Layout::from_size_align(65536, 65536)?;
        let src = alloc_zeroed(layout);
        let dst = alloc_zeroed(layout);
        if src.is_null() || dst.is_null() {
            return Err("allocation failed".into());
        }
        for i in 0..65536 {
            src.add(i).write((i % 251) as u8);
        }
        let mut source_cookie = 0;
        let mut dest_cookie = 0;
        check(sdma::sdma_pin_umem(
            fd,
            src.cast(),
            65536,
            &mut source_cookie,
        ))?;
        check(sdma::sdma_pin_umem(fd, dst.cast(), 65536, &mut dest_cookie))?;
        let channel = sdma::sdma_alloc_chn(fd);
        if channel.is_null() {
            return Err("no exclusive SDMA channel available".into());
        }
        let mut task = sdma::Task {
            src_addr: src as u64,
            dst_addr: dst as u64,
            src_process_id: pid,
            dst_process_id: pid,
            length: 65536,
            ..Default::default()
        };
        check(sdma::sdma_copy_data(channel, &mut task, 1))?;
        check(sdma::sdma_wait_chn(channel, 1))?;
        std::arch::asm!("dsb sy", options(nostack, preserves_flags));
        assert_eq!(
            std::slice::from_raw_parts(src, 65536),
            std::slice::from_raw_parts(dst, 65536)
        );
        check(sdma::sdma_free_chn(channel))?;
        check(sdma::sdma_unpin_umem(fd, dest_cookie))?;
        check(sdma::sdma_unpin_umem(fd, source_cookie))?;
        dealloc(dst, layout);
        dealloc(src, layout);
        println!("PASS: SDMA device {device}, exclusive channel, 65536 bytes exact");
    }
    Ok(())
}
#[cfg(not(all(feature = "sdma", target_arch = "aarch64", target_os = "linux")))]
fn main() {
    panic!("sdma_smoke requires --features sdma on Linux aarch64 hardware");
}
