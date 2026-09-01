from ultralytics import settings

# View all settings
print(settings)

# Return a specific setting
value = settings["runs_dir"]


# Update a setting
#settings.update({"runs_dir": "/path/to/runs"})

# Update multiple settings
#settings.update({"runs_dir": "/path/to/runs", "tensorboard": False})

# Reset settings to default values
#settings.reset()
