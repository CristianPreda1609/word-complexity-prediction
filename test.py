import torch
print(torch.cuda.is_available())
print(torch.cuda.current_device()) # Va da eroare dacă is_available() e False
print(torch.cuda.get_device_name(0)) # Va da eroare dacă is_available() e False