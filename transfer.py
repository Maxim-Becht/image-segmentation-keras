from keras_segmentation.models.all_models import model_from_name
from keras_segmentation.models.model_utils import transfer_weights

model_name = 'vgg_segnet'
n_classes_pretrained = 23
n_classes_new = 4
input_height = 480
input_width = 640

latest_checkpoint = 'data/trained/SegNet_VGG16_STIHL/v_1/.00180'


#pretrained_model = model_from_name[model_name](
#                n_classes_pretrained, input_height=input_height, input_width=input_width)
#pretrained_model.load_weights(latest_checkpoint)

new_model = model_from_name[model_name](
                n_classes_new, input_height=input_height, input_width=input_width)

#transfer_weights(pretrained_model, new_model)

new_model.train(
    train_images = "data/datasets/STIHL_SemSeg_Data/color_masks_simplyfied_3/train/raw/",
    train_annotations = "data/datasets/STIHL_SemSeg_Data/color_masks_simplyfied_3/train/labeled/",
    checkpoints_path = "data/trained/SegNet_VGG16_STIHL_simple_3/v_1/" , 
    epochs=5,
    ignore_zero_class=True,
    auto_resume_checkpoint=True
)


# model.train(
#     train_images =  "dataset1/images_prepped_train/",
#     train_annotations = "dataset1/annotations_prepped_train/",
#     checkpoints_path = "/tmp/vgg_unet_1" , epochs=5
# )

# out = model.predict_segmentation(
#     inp="dataset1/images_prepped_test/0016E5_07965.png",
#     out_fname="/tmp/out.png"
# )

# import matplotlib.pyplot as plt
# plt.imshow(out)

# # evaluating the model 
print(new_model.evaluate_segmentation( inp_images_dir="data/datasets/STIHL_SemSeg_Data/color_masks_simplyfied_3/test/raw/"  , annotations_dir="data/datasets/STIHL_SemSeg_Data/color_masks_simplyfied_3/test/labeled/" ))