{
  description = "Flake for fused RMS normalization kernel";

  inputs = {
    kernel-builder.url = "github:ChipFlow/kernels/metal-stack";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
