use crossbeam_channel::Receiver;
use std::io::{self, BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};

pub(crate) struct NodeRuntime {
    stdin: ChildStdin,
    stdout_recv: Receiver<String>,
    process: Child,
}

impl std::fmt::Debug for NodeRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeRuntime").finish_non_exhaustive()
    }
}

impl NodeRuntime {
    pub fn new() -> Result<Self, io::Error> {
        let mut process = Command::new("node")
            .arg("repl.js")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;

        let stdin = process.stdin.take().unwrap();
        let (stdout_send, stdout_recv) = crossbeam_channel::unbounded();

        let stdout = process.stdout.take().unwrap();
        std::thread::spawn(move || {
            stdout_send.send(String::new()).unwrap(); // signal ready
            let stdout = BufReader::new(stdout);
            for line in stdout.lines() {
                if let Err(e) = stdout_send.send(line.unwrap()) {
                    println!("js_runtime send error {:?}", e);
                    break;
                }
            }
        });

        // await ready
        assert_eq!(stdout_recv.recv().unwrap(), String::new());

        Ok(Self {
            stdin,
            stdout_recv,
            process,
        })
    }

    /// Block for the first output line, then emit the rest non-blocking
    pub fn output(&mut self) -> impl Iterator<Item = String> + '_ {
        let first = self.stdout_recv.recv().unwrap();
        let next = self.stdout_recv.try_iter();
        std::iter::once(first).chain(next)
    }

    pub fn eval(&mut self, code: &str) -> Result<(), io::Error> {
        self.stdin.write_all(code.as_bytes())
    }

    pub fn eval_file(&mut self, filename: &str) -> Result<(), io::Error> {
        let cmd = format!(".load {filename}\n");
        self.stdin.write_all(cmd.as_bytes())
    }
}

impl Drop for NodeRuntime {
    fn drop(&mut self) {
        self.process.kill().unwrap()
    }
}
