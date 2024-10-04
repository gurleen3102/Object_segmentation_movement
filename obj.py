import argparse
import torch
from PIL import Image
import numpy as np
from torchvision import models, transforms
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
import cv2

# Function to load and preprocess the image
def preprocess_image(image_path):
    print(f"Loading and preprocessing image: {image_path}")
    input_image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize image to a fixed size for the model
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image).unsqueeze(0)
    original_size = input_image.size  # Save original size for resizing later
    return input_tensor, original_size, np.array(input_image)

# Function to perform segmentation using DeepLabV3 model
def perform_segmentation(input_tensor):
    print("Loading DeepLabV3 model and performing segmentation...")
    model = models.segmentation.deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1).eval()
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()
    return output_predictions

# Function to create a red mask for the segmented object
def create_mask(image_with_mask, output_predictions, original_size):
    # Highlight the largest segmented object
    unique_classes, counts = np.unique(output_predictions, return_counts=True)
    most_common_class = unique_classes[np.argmax(counts[1:])]  # Ignore background

    # Create a mask for the most prominent object
    object_mask = np.zeros_like(output_predictions)
    object_mask[output_predictions == most_common_class] = 255

    # Resize the object mask back to the original image size
    object_mask_resized = cv2.resize(object_mask, original_size, interpolation=cv2.INTER_NEAREST)

    # Create the red mask overlay for the segmented region
    red_mask = np.zeros_like(image_with_mask)
    red_mask[..., 0] = object_mask_resized  # Red channel for the mask

    # Only apply the mask to the segmented part, leave the rest of the image untouched
    masked_image = np.copy(image_with_mask)
    masked_image[object_mask_resized == 255] = cv2.addWeighted(image_with_mask[object_mask_resized == 255], 1, red_mask[object_mask_resized == 255], 0.5, 0)

    return masked_image, object_mask_resized

# Function to move the segmented object in the image
def move_object(masked_image, object_mask_resized, x_shift, y_shift):
    print(f"Moving object by x_shift={x_shift} and y_shift={y_shift}")
    contours, _ = cv2.findContours(object_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        object_roi = masked_image[y:y+h, x:x+w]
        masked_image[y:y+h, x:x+w] = [0, 0, 0]  # Remove object from the original position
        new_x, new_y = x + x_shift, y + y_shift
        masked_image[new_y:new_y+h, new_x:new_x+w] = object_roi  # Place object in new position
    return masked_image

# Main function to call the segmentation and movement tasks
def segment_and_move_object(image_path, object_name, output_path, x_shift=0, y_shift=0):
    try:
        # Preprocess image
        input_tensor, original_size, image_with_mask = preprocess_image(image_path)

        # Perform segmentation
        output_predictions = perform_segmentation(input_tensor)

        # Create mask and overlay
        masked_image, object_mask_resized = create_mask(image_with_mask, output_predictions, original_size)

        # Move the object if shifts are provided
        if x_shift != 0 or y_shift != 0:
            masked_image = move_object(masked_image, object_mask_resized, x_shift, y_shift)

        # Save the output image
        if output_path:
            cv2.imwrite(output_path, masked_image)
            print(f"Output saved to {output_path}")
        else:
            print("No output path specified, skipping saving image.")
        
        return masked_image

    except Exception as e:
        print(f"Error during segmentation: {e}")
        return None

if __name__ == "__main__":
    # Argument parser for user inputs
    parser = argparse.ArgumentParser(description="Segment and move objects in an image based on user-defined class.")
    parser.add_argument('--image', required=True, help='Path to the input image')
    parser.add_argument('--object', required=True, help='Class name of the object to segment (e.g., "shelf")')
    parser.add_argument('--output', required=True, help='Path to save the output image with segmentation')
    parser.add_argument('--x', type=int, required=False, default=0, help='Shift in x direction (optional)')
    parser.add_argument('--y', type=int, required=False, default=0, help='Shift in y direction (optional)')

    args = parser.parse_args()

    # Call the segmentation and move object function
    segment_and_move_object(args.image, args.object, args.output, x_shift=args.x, y_shift=args.y)
