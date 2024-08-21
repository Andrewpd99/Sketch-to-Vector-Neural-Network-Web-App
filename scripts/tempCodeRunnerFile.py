print(torch.cuda.is_available())  # Should print True if a GPU is available
print(torch.cuda.current_device())  # Returns the index of the current GPU device being used
print(torch.cuda.get_device_name(torch.cuda.current_device()))  # Prints the name of the GPU
