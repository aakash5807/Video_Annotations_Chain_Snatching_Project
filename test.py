import cv2
import csv
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from collections import deque

# -------------------------
# 1. DEVICE
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# 2. MODEL
# -------------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = CNN().to(device)

# -------------------------
# 3. LOAD MODEL
# -------------------------
model_path = "models/epoch_5.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ Loaded model:", model_path)

# -------------------------
# 4. TRANSFORM
# -------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# -------------------------
# CSV FILES
# -------------------------
summary_csv = "results.csv"
frame_csv = "frame_annotations.csv"

# Summary CSV
with open(summary_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Video Name", "Total Frames", "Snatch Frames", "Result", "Start Frame"])

# Frame-level CSV (REAL ANNOTATION)
with open(frame_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Video", "Frame", "Label", "Probability"])

# -------------------------
# VIDEO FOLDER
# -------------------------
video_folder = "videos_test"

if not os.path.exists(video_folder):
    raise FileNotFoundError("❌ Folder not found")

video_files = [f for f in os.listdir(video_folder) if f.endswith((".mp4", ".avi", ".mov"))]

# -------------------------
# PARAMETERS
# -------------------------
SNATCH_THRESHOLD = 0.6
CONSECUTIVE_FRAMES = 5
SMOOTHING_WINDOW = 5

prob_history = deque(maxlen=SMOOTHING_WINDOW)

# -------------------------
# PROCESS VIDEOS
# -------------------------
for video_name in video_files:
    print(f"\n🎬 Processing: {video_name}")

    video_path = os.path.join(video_folder, video_name)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Cannot open {video_name}")
        continue

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"output_{video_name}", fourcc, 20.0,
                          (int(cap.get(3)), int(cap.get(4))))

    # Reset variables
    frame_count = 0
    snatch_frames = 0
    consecutive_snatch = 0
    snatch_event_detected = False
    snatch_start_frame = None
    prob_history.clear()

    # Frame loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            prob = torch.sigmoid(output).item()

        prob_history.append(prob)
        avg_prob = sum(prob_history) / len(prob_history)

        if avg_prob >= SNATCH_THRESHOLD:
            label = "SNATCH"
            color = (0, 0, 255)
            snatch_frames += 1
            consecutive_snatch += 1

            if consecutive_snatch == CONSECUTIVE_FRAMES:
                snatch_event_detected = True
                snatch_start_frame = frame_count - CONSECUTIVE_FRAMES + 1
        else:
            label = "NOT-SNATCH"
            color = (0, 255, 0)
            consecutive_snatch = 0

        # Save frame-level annotation
        with open(frame_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([video_name, frame_count, label, round(avg_prob, 3)])

        print(f"{video_name} | Frame {frame_count} → {label} ({avg_prob:.3f})")

        # Draw annotation
        cv2.putText(frame, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Save video
        out.write(frame)

        # Show video
        cv2.imshow("Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()

    cap.release()
    out.release()

    # Final result
    result_label = "SNATCH" if snatch_event_detected else "NOT-SNATCH"

    with open(summary_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            video_name,
            frame_count,
            snatch_frames,
            result_label,
            snatch_start_frame if snatch_event_detected else "-"
        ])

    print("\n========== RESULT ==========")
    print("Video:", video_name)
    print("Result:", result_label)

cv2.destroyAllWindows()

print("\n📁 Summary CSV:", os.path.abspath(summary_csv))
print("📁 Frame CSV:", os.path.abspath(frame_csv))