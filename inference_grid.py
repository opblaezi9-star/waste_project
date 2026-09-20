from collections import OrderedDict
import cv2
import torch
from torch import nn
from torchvision import models, transforms

# ---------------------------------------------------------
# 1. Load Architecture & Trained Weights
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    OrderedDict(
        [
            ("fc1", nn.Linear(2048, 512)),
            ("relu", nn.ReLU()),
            ("dropout", nn.Dropout(p=0.2)),
            ("fc2", nn.Linear(512, 2)),  # 2-class binary setup
            ("output", nn.LogSoftmax(dim=1)),
        ]
    )
)

model.load_state_dict(
    torch.load("models/waste_classifier_model.pth", map_location=device)
)
model.to(device)
model.eval()

# Validation transforms for image crops
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

# ---------------------------------------------------------
# 2. User-Defined Function for Classification
# ---------------------------------------------------------
def classify_crop(img_crop, classifier_model, compute_device, img_transform):
    """
    Takes a cropped numpy array image, processes it through the PyTorch model,
    and returns the predicted class index and text label.
    """
    tensor_img = img_transform(img_crop).unsqueeze(0).to(compute_device)
    with torch.no_grad():
        logps = classifier_model(tensor_img)
        ps = torch.exp(logps)
        _, pred_class = ps.topk(1, dim=1)

    prediction_idx = pred_class.item()  # 0: Biodegradable, 1: Non-Biodegradable
    label = "Biodegradable" if prediction_idx == 0 else "Non-Biodegradable"
    
    return prediction_idx, label

# ---------------------------------------------------------
# 3. Initialize Live Webcam Feed
# ---------------------------------------------------------
cap = cv2.VideoCapture(0)

print("\n--- Live Grid Waste Classification Feed Started (Middle Cell [1,1] Only) ---")
print("Press 'c' to scan grid cell [1,1] | Press 'q' to quit\n")

GRID_ROWS, GRID_COLS = 3, 3  # 3x3 grid layout

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    # Get dimensions of live frame
    height, width, _ = frame.shape
    display_frame = frame.copy()
    cell_h = height // GRID_ROWS
    cell_w = width // GRID_COLS

    # Draw faint reference grid lines on live feed window
    for r in range(1, GRID_ROWS):
        cv2.line(display_frame, (0, r * cell_h), (width, r * cell_h), (200, 200, 200), 1)
    for c in range(1, GRID_COLS):
        cv2.line(display_frame, (c * cell_w, 0), (c * cell_w, height), (200, 200, 200), 1)

    # Show live webcam feed with grid overlay
    cv2.imshow("Live Waste Sorting Grid", display_frame)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):  # Press 'c' to scan only the middle [1,1] grid cell
        print("\n[Action] Scanning middle grid cell [1,1] only...")
        
        r, c = 1, 1  # Middle cell coordinates
        x1 = c * cell_w
        y1 = r * cell_h
        x2 = x1 + cell_w
        y2 = y1 + cell_h

        # Slice grid cell from the captured frame
        cell_img = frame[y1:y2, x1:x2]

        if cell_img.size > 0:
            # ---------------------------------------------------------
            # Call the user-defined function here
            # ---------------------------------------------------------
            prediction, label_text = classify_crop(cell_img, model, device, transform)
            
            print(f"Grid Cell [1,1] classified as: {label_text}")
            
            # Draw visual indicator on display image (Green = Bio, Red = Non-Bio)
            box_color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                display_frame,
                f"[1,1] {label_text[:3]}",
                (x1 + 10, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                2,
            )

        print("Scan complete. Ready for next capture.\n")
        
        # Refresh display window to show the classification box for [1,1]
        cv2.imshow("Live Waste Sorting Grid", display_frame)
        cv2.waitKey(1000)  # Pause for 1 second to view the bounding box

    elif key == ord('q'):  # Press 'q' to exit
        print("Exiting application...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()