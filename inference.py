import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
  def __init__(self, input_size=7, h1=64, h2=32, h3=16, output_size=1):
    super().__init__()
    self.fc1=nn.Linear(input_size,h1)
    self.fc2=nn.Linear(h1,h2)
    self.fc3=nn.Linear(h2,h3)
    self.output=nn.Linear(h3,output_size)
  def forward(self,X):
    x=F.relu(self.fc1(X))
    x=F.relu(self.fc2(x))
    x=F.relu(self.fc3(x))
    x=self.output(x)
    return x

checkpoint = torch.load("models/car_price_predictor.pth",weights_only=False)
model=Model()
model.load_state_dict(checkpoint["model_state_dict"])

x_scaler=checkpoint["x_scaler"]
y_scaler=checkpoint["y_scaler"]

model.eval()

def predict_price(brand_id, type_id, transmission_type,fuel_type,year,kilometers, owner):
    with torch.no_grad():
        inp=[[brand_id, type_id, transmission_type,fuel_type,year,kilometers, owner]]
        inp=x_scaler.transform(inp)
        inp=torch.FloatTensor(inp)
        pred=model(inp)
        pred=y_scaler.inverse_transform(pred)
        price=round(pred[0][0])
        return price
    return None

def initiate_prediction(brand,car_type,transmission,fuel,owner, year,kilometers):
    # Process
    brands={'Mahindra': 0, 'Skoda': 1, 'Maruti Suzuki': 2, 'Hyundai': 3, 'MG': 4, 'Audi': 5, 'Toyota': 6, 'Honda': 7, 'Tata': 8, 'Ford': 9, 'Chevrolet': 10, 'BMW': 11, 'Volkswagen': 12, 'Jaguar': 13, 'Renault': 14, 'Kia': 15, 'Range Rover': 16, 'Nissan': 17}
    types={'SUV': 0, 'Sedan': 1, 'Hatchback': 2, 'MPV': 3, 'Luxury': 4}
    transmissions={'Manual': 0, 'Automatic': 1}
    fuels={'CNG': 0, 'Petrol': 1, 'Diesel': 2, 'Electric': 3, 'Hybrid': 4}
    owners={'1st': 0, '3rd+': 1, '2nd': 2}
    brand_id=brands[brand]
    type_id=types[car_type]
    transmission_type_id=transmissions[transmission]
    fuel_type_id=fuels[fuel]
    owner_id=owners[owner]
    return predict_price(brand_id, type_id, transmission_type_id,fuel_type_id,year,kilometers, owner_id)