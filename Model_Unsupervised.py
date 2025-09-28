import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ThetaConstraint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta_raw, gamma, max_iter):
        gamma_clamped = torch.clamp(gamma, min=1.05, max=2.0)
        theta_max = theta_raw[0]
        exponents = -torch.arange(max_iter, device=theta_raw.device)
        theta_new = theta_max * (gamma_clamped ** exponents)
        ctx.save_for_backward(theta_raw, gamma_clamped, exponents)
        return theta_new

    @staticmethod
    def backward(ctx, grad_output):
        theta_raw, gamma_clamped, exponents = ctx.saved_tensors
        grad_theta_raw = torch.zeros_like(theta_raw)
        grad_theta_raw[0] = torch.sum(grad_output * (gamma_clamped ** exponents))
        grad_gamma = torch.sum(
            grad_output * theta_raw[0] * exponents * (gamma_clamped ** (exponents - 1))
        )
        grad_gamma = torch.clamp(grad_gamma, min=-0.1, max=0.1)
        return grad_theta_raw, grad_gamma, None


class AdaLISTA(torch.nn.Module):
    def __init__(self, n_features, n_atoms, max_iter=100, lambd=1.0):
        super(AdaLISTA, self).__init__()
        self.n_features = n_features
        self.n_atoms = n_atoms
        self.max_iter = max_iter
        self.lambd = lambd

        self.W1_list = torch.nn.ParameterList()
        self.W2_list = torch.nn.ParameterList()
        self.last_iter = 0
        self.scale = torch.nn.Parameter(torch.tensor(1.0, device=device))

        self.theta_raw = torch.nn.Parameter(torch.zeros(1, device=device))
        self.gamma = torch.nn.Parameter(torch.tensor(1.5, device=device))
        self.gamma.register_hook(lambda grad: torch.clamp(grad, min=-0.1, max=0.1))
        self.register_buffer('theta_initialized', torch.tensor(False, device=device))
        self.register_buffer('theta_constrained', torch.zeros(max_iter, device=device))
        self.theta_max_scale = 0.1

        for _ in range(max_iter):
            W1 = torch.nn.Parameter(torch.eye(n_features, device=device), requires_grad=True)
            W2 = torch.nn.Parameter(torch.eye(n_features, device=device), requires_grad=True)
            self.W1_list.append(W1)
            self.W2_list.append(W2)

        self.current_iter = 0

    def forward(self, y, D):
        batch_size = y.shape[1]
        x = torch.zeros(self.n_atoms, batch_size, device=y.device)
        D = D.to(y.device)
        D = D.to(self.W1_list[0].dtype)

        if not self.theta_initialized:
            with torch.no_grad():
                Dt_y = torch.matmul(D.t(), y)
                theta_max = torch.max(torch.abs(Dt_y))
                theta_max = theta_max * self.theta_max_scale
                self.theta_raw.data.copy_(theta_max)
                L = torch.norm(D, p=2) ** 2
                self.gamma.data.copy_(1.1)
                self.gamma.data.clamp_(min=1.01, max=2.0)
            self.theta_initialized.data = torch.tensor(True)

        theta_constrained = ThetaConstraint.apply(self.theta_raw, self.gamma, self.max_iter)
        self.theta_constrained = theta_constrained

        prev_x = None
        stop_counter = 0
        tolerance = 0.1
        self.current_iter = 0

        while self.current_iter < self.max_iter:
            W1_k = self.W1_list[self.current_iter]
            W2_k = self.W2_list[self.current_iter]
            theta_k = theta_constrained[self.current_iter]
            theta_k = torch.clamp(theta_k)
            identity = torch.eye(self.n_atoms, device=y.device)

            W2D = torch.matmul(W2_k, D)
            DT_W2T = torch.matmul(D.t(), W2_k.t())
            term1_part = torch.matmul(DT_W2T, W2D)
            term1_coeff = identity - self.gamma * term1_part
            term1 = torch.matmul(term1_coeff, x)

            W1y = torch.matmul(W1_k.t(), y)
            DT_W1y = torch.matmul(D.t(), W1y)
            term2 = self.gamma * DT_W1y

            combined = term1 + term2

            x_new = torch.where(combined > theta_k, combined - theta_k,
                                torch.where(combined < -theta_k, combined + theta_k,
                                            torch.zeros_like(combined, device=y.device)))

            if prev_x is not None:
                diff = torch.norm(x_new - prev_x).item()
                if diff < tolerance:
                    stop_counter += 1
                    if stop_counter >= 2:
                        x = x_new
                        break
                else:
                    stop_counter = 0

            prev_x = x_new.clone()
            x = x_new
            self.current_iter += 1

        x = x * self.scale
        self.last_iter = self.current_iter

        return x