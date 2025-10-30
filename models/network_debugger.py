import torch

class NetworkDebugger:
    def __init__(self, model):
        self.model = model
        self.handles = []

    def register_hooks(self):
        def forward_hook(module, inputs, outputs):
            if isinstance(outputs, torch.Tensor):
                print(f"[FORWARD] {module.__class__.__name__:<20} "
                      f"→ shape={tuple(outputs.shape)}, "
                      f"mean={outputs.mean():.4f}, std={outputs.std():.4f}")

        def backward_hook(module, grad_in, grad_out):
            grad = grad_out[0]
            if isinstance(grad, torch.Tensor):
                print(f"[BACKWARD] {module.__class__.__name__:<20} "
                      f"grad_mean={grad.mean():.4f}, grad_std={grad.std():.4f}")

        for m in self.model.modules():
            if len(list(m.children())) == 0:  # leaf module only
                self.handles.append(m.register_forward_hook(forward_hook))
                self.handles.append(m.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
