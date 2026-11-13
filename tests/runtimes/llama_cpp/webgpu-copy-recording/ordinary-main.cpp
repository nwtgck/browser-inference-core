int main() {
    host_success_and_lifetime();
    host_guards();
    gpu_order_guards_and_fallback();
    scheduler_and_cpu_mode();
    errors();
    std::cout << "PASS: " << checks
              << " actual-helper ordering/lifetime/guard/fallback/scheduler checks (deferred CPU queue)\n";
}
