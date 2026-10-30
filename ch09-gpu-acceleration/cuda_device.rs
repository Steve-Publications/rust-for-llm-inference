let device = Device::new_cuda(0)?;
let tensor = Tensor::randn(0.0, 1.0, (1, 768, 768), &device)?;
