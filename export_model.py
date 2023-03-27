import torch
import conv1d_classifier


if __name__ == "__main__":
    modelpath = "./models/ahgtrackmodel9.pth"
    model = conv1d_classifier.Conv1dClassifier(0)
    model.load_state_dict(torch.load(modelpath))
    traced = torch.jit.trace(model, torch.rand(1, 1033))
    traced.save("ahgtrackmodel9.pt")
