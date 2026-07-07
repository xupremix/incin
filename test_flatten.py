with open("crates/kindle/examples/mnist_training.rs", "r") as f:
    content = f.read()

content = content.replace(
    "Linear::<Dyn, Backend>::new_dyn((784, 128))?,",
    "Linear::<s![784, 128], Backend>::new(),"
)

with open("crates/kindle/examples/mnist_training.rs", "w") as f:
    f.write(content)
