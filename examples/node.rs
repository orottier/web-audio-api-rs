use web_audio_api::js_runtime::NodeRuntime;

fn main() -> std::io::Result<()> {
    println!("CWD: {:?}", std::env::current_dir().unwrap());

    let mut runtime = NodeRuntime::new()?;
    runtime.eval_file("test.js")?;
    runtime.output().for_each(|o| println!("{o}"));

    let code = "
let inputs = [[[]]];
let outputs = [[[0.0]]];
let parameters = {};
proc.process(inputs, outputs, parameters);
console.log(outputs);
console.log('Done123');
";
    runtime.eval(code)?;
    'outer: loop {
        for o in runtime.output() {
            println!("{o}");
            if o.contains("> Done123") {
                break 'outer;
            }
        }
    }

    Ok(())
}
