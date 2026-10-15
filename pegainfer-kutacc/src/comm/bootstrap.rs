//! Bounded, rank-ordered TCP bootstrap for KuPL and KURMCL OOB callbacks.
//! Use a separate endpoint and cookie per communicator. Peers must be trusted;
//! the cookie detects accidental cross-job connections, not hostile clients.
use std::ffi::c_void;
use std::io::Read;
use std::io::Write;
use std::io::{self};
use std::net::SocketAddr;
use std::net::TcpListener;
use std::net::TcpStream;
use std::time::Duration;
use std::time::Instant;

const MAX_BYTES: usize = 64 * 1024 * 1024;

/// One thread's bootstrap domain. Keep the Box stable while native callbacks
/// refer to it; native collective calls must not overlap on this domain.
pub struct Bootstrap {
    rank: usize,
    world: usize,
    streams: Vec<TcpStream>,
    sequence: u64,
    poisoned: bool,
    error: String,
    // Do not allow runtime handles to migrate between Rust threads.
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
fn configure(s: &TcpStream, timeout: Duration) -> io::Result<()> {
    s.set_nodelay(true)?;
    s.set_read_timeout(Some(timeout))?;
    s.set_write_timeout(Some(timeout))
}
fn read_u64(s: &mut TcpStream) -> io::Result<u64> {
    let mut b = [0; 8];
    s.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
impl Bootstrap {
    /// Connect a bootstrap group. Rank zero must listen on `address`; other ranks
    /// use that same reachable address. Rank/world/cookie are checked on connect.
    pub fn connect(
        address: SocketAddr,
        rank: usize,
        world: usize,
        cookie: u64,
        timeout: Duration,
    ) -> io::Result<Box<Self>> {
        if world == 0 || world > 256 || rank >= world || timeout.is_zero() {
            return Err(invalid("invalid rank/world/timeout"));
        }
        let deadline = Instant::now() + timeout;
        let mut streams = Vec::new();
        if rank == 0 {
            let listener = TcpListener::bind(address)?;
            listener.set_nonblocking(true)?;
            let mut peers: Vec<Option<TcpStream>> = (0..world).map(|_| None).collect();
            for _ in 1..world {
                let mut s = loop {
                    match listener.accept() {
                        Ok((s, _)) => break s,
                        Err(e)
                            if e.kind() == io::ErrorKind::WouldBlock
                                && Instant::now() < deadline =>
                        {
                            std::thread::sleep(Duration::from_millis(5))
                        }
                        Err(e) => return Err(e),
                    }
                };
                configure(&s, timeout)?;
                let c = read_u64(&mut s)?;
                let n = read_u64(&mut s)?;
                let r = read_u64(&mut s)? as usize;
                if c != cookie || n != world as u64 || r == 0 || r >= world || peers[r].is_some() {
                    return Err(invalid("bootstrap identity mismatch"));
                }
                s.write_all(&cookie.to_le_bytes())?;
                peers[r] = Some(s);
            }
            streams = peers.into_iter().skip(1).map(Option::unwrap).collect();
        } else {
            let mut s = loop {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    return Err(io::Error::new(io::ErrorKind::TimedOut, "bootstrap connect"));
                }
                match TcpStream::connect_timeout(
                    &address,
                    remaining.min(Duration::from_millis(100)),
                ) {
                    Ok(s) => break s,
                    Err(_) => std::thread::sleep(Duration::from_millis(5)),
                }
            };
            configure(&s, timeout)?;
            for v in [cookie, world as u64, rank as u64] {
                s.write_all(&v.to_le_bytes())?;
            }
            if read_u64(&mut s)? != cookie {
                return Err(invalid("bootstrap cookie mismatch"));
            }
            streams.push(s);
        }
        Ok(Box::new(Self {
            rank,
            world,
            streams,
            sequence: 0,
            poisoned: false,
            error: String::new(),
            _thread: std::marker::PhantomData,
        }))
    }
    pub fn rank(&self) -> usize {
        self.rank
    }
    pub fn world(&self) -> usize {
        self.world
    }
    pub fn last_error(&self) -> &str {
        &self.error
    }
    pub fn context(&mut self) -> *mut c_void {
        (self as *mut Self).cast()
    }

    fn exchange(&mut self, op: u64, send: &[u8], recv: &mut [u8]) -> io::Result<()> {
        if self.poisoned {
            return Err(invalid("bootstrap domain poisoned"));
        }
        let expected = if op == 2 {
            send.len()
        } else {
            send.len()
                .checked_mul(self.world)
                .ok_or_else(|| invalid("size overflow"))?
        };
        if recv.len() != expected
            || send
                .len()
                .checked_mul(self.world)
                .is_none_or(|v| v > MAX_BYTES)
            || (op == 2 && !send.len().is_multiple_of(self.world))
        {
            return Err(invalid("bootstrap message too large or mismatched"));
        }
        let result = self.exchange_inner(op, send, recv);
        if let Err(ref e) = result {
            self.poisoned = true;
            self.error = e.to_string();
        }
        result
    }
    fn exchange_inner(&mut self, op: u64, send: &[u8], recv: &mut [u8]) -> io::Result<()> {
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or_else(|| invalid("sequence overflow"))?;
        let header = [self.sequence, op, send.len() as u64];
        if self.rank == 0 {
            let mut all = vec![0; send.len() * self.world];
            all[..send.len()].copy_from_slice(send);
            for (i, s) in self.streams.iter_mut().enumerate() {
                for v in header {
                    if read_u64(s)? != v {
                        return Err(invalid("bootstrap collective order/shape mismatch"));
                    }
                }
                s.read_exact(&mut all[(i + 1) * send.len()..(i + 2) * send.len()])?;
            }
            if op == 2 {
                let chunk = send.len() / self.world;
                for dest in 0..self.world {
                    for src in 0..self.world {
                        recv[src * chunk..(src + 1) * chunk].copy_from_slice(
                            &all[src * send.len() + dest * chunk
                                ..src * send.len() + (dest + 1) * chunk],
                        );
                    }
                    if dest > 0 {
                        self.streams[dest - 1].write_all(recv)?;
                    }
                }
                for src in 0..self.world {
                    recv[src * chunk..(src + 1) * chunk]
                        .copy_from_slice(&all[src * send.len()..src * send.len() + chunk]);
                }
            } else {
                recv.copy_from_slice(&all);
                for s in &mut self.streams {
                    s.write_all(recv)?;
                }
            }
        } else {
            let s = &mut self.streams[0];
            for v in header {
                s.write_all(&v.to_le_bytes())?;
            }
            s.write_all(send)?;
            s.read_exact(recv)?;
            if read_u64(s)? != self.sequence {
                return Err(invalid("bootstrap completion sequence"));
            }
        }
        if self.rank == 0 {
            for s in &mut self.streams {
                s.write_all(&self.sequence.to_le_bytes())?;
            }
        }
        Ok(())
    }
    pub fn allgather(&mut self, send: &[u8], recv: &mut [u8]) -> io::Result<()> {
        self.exchange(1, send, recv)
    }
    pub fn alltoall(&mut self, send: &[u8], recv: &mut [u8]) -> io::Result<()> {
        self.exchange(2, send, recv)
    }
    pub fn barrier(&mut self) -> io::Result<()> {
        self.exchange(3, &[], &mut [])
    }
}

fn bytes(count: i32, datatype: i32) -> Option<usize> {
    let width = match datatype {
        0 => 1,
        1 | 3 => 4,
        2 | 4 => 8,
        _ => return None,
    };
    usize::try_from(count).ok()?.checked_mul(width)
}

/// Native OOB callback table, layout-compatible with kurmcl_oob_cb_t.
#[repr(C)]
pub struct RdmaCallbacks {
    pub allgather: unsafe extern "C" fn(*const c_void, *mut c_void, i32, *mut c_void, i32) -> i32,
    pub barrier: unsafe extern "C" fn(*mut c_void) -> i32,
    pub alltoall:
        unsafe extern "C" fn(*const c_void, i32, i32, *mut c_void, i32, i32, *mut c_void) -> i32,
}
pub const RDMA_CALLBACKS: RdmaCallbacks = RdmaCallbacks {
    allgather: oob_allgather,
    barrier: oob_barrier,
    alltoall: oob_alltoall,
};

// Callbacks catch panics: Rust unwinding must never cross the native ABI.
fn callback(f: impl FnOnce() -> io::Result<()>) -> i32 {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(Ok(())) => 0,
        _ => -1,
    }
}
/// # Safety
/// `group` is a live exclusive Bootstrap; buffers cover the datatype/count and
/// world. A null send pointer requests in-place allgather from recv[rank].
pub unsafe extern "C" fn oob_allgather(
    send: *const c_void,
    recv: *mut c_void,
    count: i32,
    group: *mut c_void,
    datatype: i32,
) -> i32 {
    callback(|| {
        if group.is_null() {
            return Err(invalid("null group"));
        }
        let g = unsafe { &mut *group.cast::<Bootstrap>() };
        let n = bytes(count, datatype).ok_or_else(|| invalid("datatype/count"))?;
        let total = n
            .checked_mul(g.world)
            .filter(|v| *v <= MAX_BYTES)
            .ok_or_else(|| invalid("size"))?;
        if total == 0 {
            return g.allgather(&[], &mut []);
        }
        if recv.is_null() {
            return Err(invalid("null recv"));
        }
        let src = if send.is_null() {
            unsafe { recv.cast::<u8>().add(g.rank * n) }
        } else {
            send.cast::<u8>()
        };
        let owned = unsafe { std::slice::from_raw_parts(src, n) }.to_vec();
        g.allgather(&owned, unsafe {
            std::slice::from_raw_parts_mut(recv.cast(), total)
        })
    })
}
/// # Safety
/// `group` is a live exclusive Bootstrap used by every member of the group.
pub unsafe extern "C" fn oob_barrier(group: *mut c_void) -> i32 {
    callback(|| {
        if group.is_null() {
            return Err(invalid("null group"));
        }
        unsafe { &mut *group.cast::<Bootstrap>() }.barrier()
    })
}
/// # Safety
/// Same group contract as allgather; both buffers cover world * per-peer count.
pub unsafe extern "C" fn oob_alltoall(
    send: *const c_void,
    sc: i32,
    st: i32,
    recv: *mut c_void,
    rc: i32,
    rt: i32,
    group: *mut c_void,
) -> i32 {
    callback(|| {
        if group.is_null() {
            return Err(invalid("null group"));
        }
        let g = unsafe { &mut *group.cast::<Bootstrap>() };
        let n = bytes(sc, st).ok_or_else(|| invalid("send datatype/count"))?;
        if bytes(rc, rt) != Some(n) {
            return Err(invalid("alltoall send/recv mismatch"));
        }
        let total = n
            .checked_mul(g.world)
            .filter(|v| *v <= MAX_BYTES)
            .ok_or_else(|| invalid("size"))?;
        if total == 0 {
            return g.alltoall(&[], &mut []);
        }
        if send.is_null() || recv.is_null() {
            return Err(invalid("null buffer"));
        }
        let owned = unsafe { std::slice::from_raw_parts(send.cast::<u8>(), total) }.to_vec();
        g.alltoall(&owned, unsafe {
            std::slice::from_raw_parts_mut(recv.cast(), total)
        })
    })
}
