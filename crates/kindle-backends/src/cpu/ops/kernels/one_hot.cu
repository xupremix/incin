extern "C" __global__ void build_one_hot(
    const long long* __restrict__ targets,
    float* __restrict__ one_hot,
    int batch,
    int classes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch) return;

    long long class_idx = targets[i];
    if (class_idx >= 0 && class_idx < classes) {
        one_hot[i * classes + class_idx] = 1.0f;
    }
}
