trait AsMeh {
    type C;
    fn meh(c: Self::C) -> String;
}

fn meh_me<T: AsMeh>(c: T::C) -> String {
    T::meh(c)
}

struct MehImpl;

impl AsMeh for MehImpl {
    type C = usize;
    fn meh(c: Self::C) -> String {
        format!("{}", c)
    }
}

fn main() {
    println!("{}", meh_me::<MehImpl>(13));
}
