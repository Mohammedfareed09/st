import sys, os, inspect

# Ensure Python can find fairseq_signals
sys.path.append(os.path.join(os.getcwd(), "fairseq-signals"))

from fairseq_signals.models import build_model_from_checkpoint

print("Signature of build_model_from_checkpoint:")
print(inspect.signature(build_model_from_checkpoint))
