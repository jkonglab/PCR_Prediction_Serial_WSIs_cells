from utils import DataGenerator, read_annotation_lines
from models import Yolov4
from config import yolo_config
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint


early_stopper = EarlyStopping(min_delta=0.001, patience=10)


train_lines, val_lines = read_annotation_lines('./dataset/txt/anno.txt', test_size=0.1)
FOLDER_PATH = './dataset/img'
class_name_path = './class_names/coco_classes.txt'
data_gen_train = DataGenerator(train_lines, class_name_path, FOLDER_PATH)
data_gen_val = DataGenerator(val_lines, class_name_path, FOLDER_PATH)

model = Yolov4(weight_path='../yolov4.weights', 
               class_name_path=class_name_path)

checkpoint_filepath = './DetectorCheckpoints/'
model_checkpoint_callback = ModelCheckpoint(
	filepath=checkpoint_filepath+'model.{epoch:03d}-{val_loss:.3f}-{val_acc:.3f}.h5',
	monitor='val_loss',
	mode='min')


model.fit(data_gen_train, 
          initial_epoch=0,
          epochs=10000, 
          val_data_gen=data_gen_val,
          callbacks=[early_stopper,model_checkpoint_callback])
