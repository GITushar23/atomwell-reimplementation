import torch
import torch.nn.functional as F

class D3PM:
    def __init__(self, vocab_size, T=100, mask_id=2):
        self.K = vocab_size
        self.T = T
        self.mask_id = mask_id

        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod

    def _get_Qt_bar(self, t, device):
        alpha_bar = self.alphas_cumprod.to(device)[t-1]
        Qt_bar = torch.zeros(len(t), self.K, self.K, device=device)
        for i in range(len(t)):
            ab = alpha_bar[i]
            # Diagonal: probability of staying at original token
            Qt_bar[i] = torch.eye(self.K, device=device) * ab
            # All tokens have (1-ab) probability of transitioning to MASK
            Qt_bar[i, :, self.mask_id] += (1 - ab)
            # This is already a valid transition matrix, no normalization needed
        return Qt_bar

    def q_sample(self, x0, t):
        B, L = x0.shape
        device = x0.device

        Qt_bar = self._get_Qt_bar(t, device)
        x0_onehot = F.one_hot(x0, self.K).float()

        qt = torch.einsum('blk,bkj->blj', x0_onehot, Qt_bar)
        x_t = torch.multinomial(qt.view(-1, self.K), 1).view(B, L)
        return x_t

    def sample_timesteps(self, batch_size, device):
        return torch.randint(1, self.T + 1, (batch_size,), device=device)

    def _get_Qt(self, t, device):
        """Single-step transition matrix Q_t"""
        if t == 0:
            return torch.eye(self.K, device=device).unsqueeze(0)

        alpha = self.alphas.to(device)[t-1]
        Qt = torch.eye(self.K, device=device) * alpha
        Qt[:, self.mask_id] += (1 - alpha)
        return Qt.unsqueeze(0)  # [1, K, K]

    def compute_posterior(self, x_t, x0, t):
        """
        Compute q(x_{t-1} | x_t, x0) - the reverse posterior (VECTORIZED)

        Using Bayes: q(x_{t-1} | x_t, x0) ∝ q(x_t | x_{t-1}) * q(x_{t-1} | x0)

        Args:
            x_t: [B, L] - current noisy tokens
            x0: [B, L] or [B, L, K] - predicted clean tokens (indices or probs)
            t: int - current timestep

        Returns:
            [B, L, K] - posterior probabilities for x_{t-1}
        """
        B, L = x_t.shape
        device = x_t.device

        if t <= 1:
            # At t=1, directly return x0
            if x0.dim() == 2:
                return F.one_hot(x0, self.K).float()
            return x0

        # Get transition matrices
        Qt = self._get_Qt(t, device)[0]  # [K, K] - q(x_t | x_{t-1})
        Qt_1_bar = self._get_Qt_bar(
            torch.tensor([t-1], device=device), device
        )[0]  # [K, K] - q(x_{t-1} | x0)

        # Convert x0 to one-hot if needed
        if x0.dim() == 2:
            x0_onehot = F.one_hot(x0, self.K).float()
        else:
            x0_onehot = x0  # Already [B, L, K]

        # VECTORIZED COMPUTATION (no loops!)
        # q(x_t | x_{t-1}): Extract columns for each x_t value
        # Qt[:, x_t[b,l]] gives q(x_t=x_t[b,l] | x_{t-1}) for all x_{t-1}

        # Gather the appropriate columns: [B, L, K]
        q_xt_given_xt_1 = Qt[:, x_t].permute(1, 2, 0)  # [B, L, K]

        # q(x_{t-1} | x0): x0_onehot @ Qt_1_bar^T
        # [B, L, K] @ [K, K] -> [B, L, K]
        q_xt_1_given_x0 = torch.matmul(x0_onehot, Qt_1_bar)  # [B, L, K]

        # Unnormalized posterior: element-wise multiplication
        posterior_unnorm = q_xt_given_xt_1 * q_xt_1_given_x0  # [B, L, K]

        # Normalize along K dimension
        posterior = posterior_unnorm / (posterior_unnorm.sum(dim=-1, keepdim=True) + 1e-8)

        return posterior

    def p_sample(self, model, x_t, t, domain, seq_boundaries, max_seqlen):
        """
        Full reverse sampling step: x_t -> x_{t-1}

        Args:
            model: The denoising model
            x_t: [B, L] - current noisy tokens
            t: int - current timestep (scalar)
            domain: [B, L] - domain labels
            seq_boundaries: list of boundary lists
            max_seqlen: int

        Returns:
            [B, L] - x_{t-1}
        """
        device = x_t.device
        B, L = x_t.shape

        # Create timestep tensor
        t_batch = torch.full((B,), t, dtype=torch.long, device=device)

        # 1) Model predicts p(x0 | x_t, t)
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                output = model(x_t, t_batch, domain,
                              seq_boundaries=seq_boundaries,
                              max_seqlen=max_seqlen)
                logits_x0 = output["logits"]  # [B, L, K]
            p_x0_given_xt = F.softmax(logits_x0.float(), dim=-1)  # [B, L, K] - convert back to fp32

        # 2) Compute posterior fully vectorized (no loops!)
        #    p(x_{t-1} | x_t) = Σ_x0 p(x0|x_t) * q(x_{t-1}|x_t,x0)

        if t <= 1:
            # At t=1, just sample from predicted x0 distribution
            x_t_minus_1 = torch.multinomial(
                p_x0_given_xt.view(-1, self.K), 1
            ).view(B, L)
            return x_t_minus_1

        # Get transition matrices
        Qt = self._get_Qt(t, device)[0]  # [K, K] - q(x_t | x_{t-1})
        Qt_1_bar = self._get_Qt_bar(
            torch.tensor([t-1], device=device), device
        )[0]  # [K, K] - q(x_{t-1} | x0)

        # Compute q(x_t | x_{t-1}) for observed x_t: [B, L, K]
        q_xt_given_xt_1 = Qt[:, x_t].permute(1, 2, 0)  # [B, L, K]

        # For each possible x0, compute q(x_{t-1} | x0)
        # Qt_1_bar[:, :] is [K_x0, K_{t-1}] - prob of x_{t-1} given x0
        # We want: for each position, marginalize over x0 weighted by p_x0_given_xt

        # Compute q(x_{t-1} | x0) for all x0 at once: [B, L, K_x0, K_{t-1}]
        # p_x0_given_xt is [B, L, K_x0]
        # Qt_1_bar is [K_x0, K_{t-1}]
        # Result: [B, L, K_{t-1}]
        q_xt_1_given_x0 = torch.einsum('blk,kj->blj', p_x0_given_xt, Qt_1_bar)  # [B, L, K]

        # Unnormalized posterior: element-wise multiplication
        posterior_unnorm = q_xt_given_xt_1 * q_xt_1_given_x0  # [B, L, K]

        # Normalize
        p_xt_1 = posterior_unnorm / (posterior_unnorm.sum(dim=-1, keepdim=True) + 1e-8)

        # 3) Sample x_{t-1} from the posterior
        x_t_minus_1 = torch.multinomial(
            p_xt_1.view(-1, self.K), 1
        ).view(B, L)

        return x_t_minus_1
