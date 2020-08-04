from simulations.scripts.create_mnist_fft import main as mnist_fft


def create_fft_images(sim_conf):
    if sim_conf['type'] == "mnist":
        mnist_fft(
            data_path=sim_conf["resource"],
            out_path=sim_conf["out_path"],
            size=sim_conf["img_size"],
            bundle_size=sim_conf["bundle_size"],
            noise=sim_conf["noise"]
        )
