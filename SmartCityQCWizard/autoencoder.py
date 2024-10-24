import torch
import torch.nn as nn
import tqdm

class AutoEncoder(nn.Module):

    def __init__(self, in_size: int, latent_size: int):
        super(AutoEncoder, self).__init__()
    
        self.encoder = nn.Sequential(
            nn.Linear(in_size, in_size // 2), nn.ReLU(),
            nn.Linear(in_size // 2, in_size // 3), nn.ReLU(),
            nn.Linear(in_size // 3, in_size // 4), nn.ReLU(),
            nn.Linear(in_size // 4, in_size // 5), nn.ReLU(),
            nn.Linear(in_size // 5, in_size // 6), nn.ReLU(),
            nn.Linear(in_size // 6, latent_size)
        )
        self.decoder = nn.Sequential(  
            nn.Linear(latent_size, in_size // 6), nn.ReLU(),
            nn.Linear(in_size // 6, in_size // 5), nn.ReLU(),
            nn.Linear(in_size // 5, in_size // 4), nn.ReLU(),
            nn.Linear(in_size // 4, in_size // 3), nn.ReLU(),
            nn.Linear(in_size // 3, in_size // 2), nn.ReLU(),
            nn.Linear(in_size // 2, in_size)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def predict(self, x):
        return self.encoder(x)

def train(model, ds, epochs=100):

    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    losses = 0
    for epoch in range(epochs):
        for data in tqdm.tqdm(dl, f"Epoch {epoch+1}/{epochs}", leave=False):
            x = data
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()
            losses += loss.item() * x.shape[0]
        # print(f"Epoch {epoch} loss: {loss.item() / len(ds)}")
    return model

@torch.no_grad()    
def predict(model, ds):
    criterion = nn.MSELoss()
    model.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
    predictions = []
    for data in tqdm.tqdm(dl, "Predicting", leave=False):
        x = data
        output = model(x)
        loss = criterion(output, x) * x.shape[0]
        predictions.append(model.predict(x))
    print(f"Mean loss: {loss / len(ds)}")
    return torch.cat(predictions, dim=0)