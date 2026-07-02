"""Upload erp3 files and run smoke test on remote."""
import paramiko, sys, os

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('i-1.gpushare.com', port=59010, username='root',
               password='HMAV6TEcCARYegYCFEGB6B89sDqqSGfU', timeout=15)

sftp = client.open_sftp()

# Upload modified files
files = [
    'scripts/stau_opus_operator.py',
    'scripts/patch_whisper_stau.py',
    'scripts/patch_whisper_sparse_grad.py',
    'scripts/train_erp3.py',
]

for f in files:
    local = os.path.join(r'F:\τ\点心杯', f)
    remote = f'/hy-tmp/dimsum/{f}'
    sftp.put(local, remote)
    print(f'  uploaded: {f}')

sftp.close()

# Run smoke test
print('\n=== Smoke test: τ-opus + SparseGrad forward pass ===')
stdin, stdout, stderr = client.exec_command(
    'cd /hy-tmp/dimsum && python3 -c "\n'
    'import os, sys, torch\n'
    'os.environ["HF_HOME"] = "/hy-tmp/hf_cache"\n'
    'sys.path.insert(0, "/hy-tmp/dimsum/scripts")\n'
    'from transformers import WhisperForConditionalGeneration\n'
    'from patch_whisper_stau import apply_learnable_stau, remove_stau\n'
    'from patch_whisper_sparse_grad import apply_sparse_grad_wta\n'
    '\n'
    'model = WhisperForConditionalGeneration.from_pretrained(\n'
    '    "/hy-tmp/whisper-small-local",\n'
    '    attn_implementation="eager",\n'
    '    torch_dtype=torch.bfloat16,\n'
    '    local_files_only=True,\n'
    ')\n'
    'apply_learnable_stau(model, tau_init=1.5, alpha_init=1.0)\n'
    'apply_sparse_grad_wta(model, k_ratio=0.3, target="encoder+decoder")\n'
    '\n'
    '# Forward pass\n'
    'feats = torch.randn(1, 80, 3000, dtype=torch.bfloat16).cuda()\n'
    'ids = torch.randint(0, model.config.vocab_size, (1, 5)).cuda()\n'
    'model = model.cuda()\n'
    'out = model(input_features=feats, decoder_input_ids=ids)\n'
    'print(f"Forward OK: logits shape = {out.logits.shape}")\n'
    '\n'
    '# Backward pass\n'
    'loss = out.logits.sum()\n'
    'loss.backward()\n'
    '\n'
    '# Check τ gradients\n'
    'tau_grads = []\n'
    'for name, m in model.named_modules():\n'
    '    if hasattr(m, "_stau_opus") and m._stau_opus is not None:\n'
    '        if m._stau_opus.log_tau.grad is not None:\n'
    '            tau_grads.append(m._stau_opus.log_tau.grad.abs().mean().item())\n'
    'print(f"τ grad mean: {sum(tau_grads)/len(tau_grads):.6f} ({len(tau_grads)} modules)")\n'
    'print("SMOKE TEST PASSED")\n'
    '" 2>&1'
)

print(stdout.read().decode())
err = stderr.read().decode()
if err:
    print("STDERR:", err)

client.close()
