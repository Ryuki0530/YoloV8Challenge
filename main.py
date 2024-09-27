import cv2
import sys
from ultralytics import YOLO

def main():
    # コマンドライン引数からIPカメラのURLまたはパスを取得
    if len(sys.argv) > 1:
        camera_source = sys.argv[1]
        print(f"IPカメラ: {camera_source} を使用。")
    else:
        # 引数がない場合はPCのWebカメラを使う
        camera_source = 0  # 0はPCのデフォルトカメラ
        print("Webカメラを使用。")


    # YOLOv8のモデルをロード
    model = YOLO('yolov8n.pt')
    # カメラの映像を取得
    cap = cv2.VideoCapture(camera_source)

    
    if not cap.isOpened():
        print(f"カメラに接続できません。: {camera_source}")
        sys.exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("映像を取得できません。")
            break

        # YOLOv8で推論を実行
        results = model(frame)

        annotated_frame = results[0].plot()

        cv2.imshow('YOLOv8 Camera Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # リソースの解放
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
