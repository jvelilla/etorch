import torch
from safetensors.torch import save_file

def main():
    n_layer = 1
    n_embd = 4
    n_head = 2
    d_ff = 4 * n_embd

    tensors = {}
    
    # We will just generate basic deterministic weights for testing
    for i in range(n_layer):
        prefix = f"blocks.{i+1}."
        tensors[prefix + "ln_1.weight"] = torch.ones(n_embd)
        tensors[prefix + "ln_1.bias"] = torch.zeros(n_embd)
        
        tensors[prefix + "attn.q_proj.weight"] = torch.eye(n_embd)
        tensors[prefix + "attn.q_proj.bias"] = torch.zeros(n_embd)
        tensors[prefix + "attn.k_proj.weight"] = torch.eye(n_embd)
        tensors[prefix + "attn.k_proj.bias"] = torch.zeros(n_embd)
        tensors[prefix + "attn.v_proj.weight"] = torch.eye(n_embd)
        tensors[prefix + "attn.v_proj.bias"] = torch.zeros(n_embd)
        tensors[prefix + "attn.out_proj.weight"] = torch.eye(n_embd)
        tensors[prefix + "attn.out_proj.bias"] = torch.zeros(n_embd)
        
        tensors[prefix + "ln_2.weight"] = torch.ones(n_embd)
        tensors[prefix + "ln_2.bias"] = torch.zeros(n_embd)
        
        tensors[prefix + "mlp_fc1.weight"] = torch.eye(n_embd, d_ff).T.contiguous()
        tensors[prefix + "mlp_fc1.bias"] = torch.zeros(d_ff)
        tensors[prefix + "mlp_fc2.weight"] = torch.eye(d_ff, n_embd).T.contiguous()
        tensors[prefix + "mlp_fc2.bias"] = torch.zeros(n_embd)

    tensors["ln_f.weight"] = torch.ones(n_embd)
    tensors["ln_f.bias"] = torch.zeros(n_embd)

    save_file(tensors, "microgpt_dummy.safetensors")
    print("microgpt_dummy.safetensors generated.")

if __name__ == "__main__":
    main()
