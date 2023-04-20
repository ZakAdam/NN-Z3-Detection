# Dataset
https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset

- images of dogs
- 120 breeds
- two main folders - each split by breed
    - Annotations - xml files with metadata and bounding boxes
    - Images - images of dogs
- each image can contain multiple dogs

# Prepare dataset

**We select only those images where is only one dog.** 

```bash
python data_preparation/prepare.py --dataset_path="data/archive.zip" --unpacked_path="data/archive" --h5path="data/dataset.h5"
```

# Run model in detach mode
Edit model in Docker container and save work. 
Leave container back to the SSH connection.
If you don't have screen already create one:
```bash
screen
```
Screen with sepcified name (optional)
```bash
screen -S session_name
```

Then you are connected to screen, where you want to start you model.
In screen you want to run this command:

```bash
docker exec -it <container_id> python3 /workspaces/cnn_exercise/src/train.py
```
Continer ID is id of your container where you are working. In my case it is: _c8e30968e1d7_. You can find this on left side, in docker extension panel. Find your container there (it has your name in volume path e.g. _/home/pato/..._). Text in bold is name of your container (probably can be used instead of ID but I haven't tried it) and next to it is container ID.

Execute commnad and it will start run model in docker container. Now, you can safely leave screen with CTRL+A+D combination. Screen will keep running in background.

In order to get back to the screen run:
```bash
screen -r
```

Or, if you have more screens then run: 
```bash
screen -r <screen_ID>
```

CTRL+A+D will detach again. You can still connect to the same screen with *screen -r* and run docker command from history.


# Train model

```bash
TBD
```

# Test model

```bash
TBD
```

# Results


TBD
