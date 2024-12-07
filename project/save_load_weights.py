import os
import torch

def save_model_weights(actor, qf1, qf2, qf1_target, qf2_target, directory="models", step=0):
    os.makedirs(directory, exist_ok=True)
    torch.save(actor.state_dict(), os.path.join(directory, f"actor_{step}.pth"))
    torch.save(qf1.state_dict(), os.path.join(directory, f"qf1_{step}.pth"))
    torch.save(qf2.state_dict(), os.path.join(directory, f"qf2_{step}.pth"))
    torch.save(qf1_target.state_dict(), os.path.join(directory, f"qf1_target_{step}.pth"))
    torch.save(qf2_target.state_dict(), os.path.join(directory, f"qf2_target_{step}.pth"))
    print(f"Models saved at step {step} to {directory}")

def save_actor_model_weights(actor, qf1, qf2, qf1_target, qf2_target, directory="models", step=0):
    os.makedirs(directory, exist_ok=True)
    torch.save(actor.state_dict(), os.path.join(directory, f"actor_{step}.pth"))
    print(f"Actor models saved at step {step} to {directory}")



def load_model_weights(actor, qf1, qf2, qf1_target, qf2_target, directory="models", step=0):
    actor.load_state_dict(torch.load(os.path.join(directory, f"actor_{step}.pth")))
    qf1.load_state_dict(torch.load(os.path.join(directory, f"qf1_{step}.pth")))
    qf2.load_state_dict(torch.load(os.path.join(directory, f"qf2_{step}.pth")))
    qf1_target.load_state_dict(torch.load(os.path.join(directory, f"qf1_target_{step}.pth")))
    qf2_target.load_state_dict(torch.load(os.path.join(directory, f"qf2_target_{step}.pth")))
    print(f"Models loaded from {directory} at step {step}")

