use robopokerlib::analysis::cli::CLI;

#[tokio::main]
async fn main() {
    // optional – enables log files / coloured output like the other binaries
    robopokerlib::logs();

    // start the interactive REPL
    CLI::run().await;
}
