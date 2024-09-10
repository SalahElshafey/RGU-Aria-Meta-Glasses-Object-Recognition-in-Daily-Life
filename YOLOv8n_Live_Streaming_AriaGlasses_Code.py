### Note you can find the code on Aria repository link https://facebookresearch.github.io/projectaria_tools/docs/ARK/sdk/samples/streaming_subscribe (in the Code Walkthrough)



import cv2
import numpy as np
from ultralytics import YOLO
from aria.sdk import DeviceClient, StreamingConfig, StreamingSecurityOptions, StreamingDataType, CameraId
import aria


model = YOLO("yolov8n.pt")  

try:
    # Connect to the device over USB
    device_client = DeviceClient()
    device = device_client.connect()

    streaming_manager = device.streaming_manager
    streaming_client = streaming_manager.streaming_client

    # Configure streaming settings with the custom profile
    streaming_config = StreamingConfig()
    streaming_config.profile_name = "profile12"  # Ensure this matches the exact name of your custom profile
    streaming_config.security_options = StreamingSecurityOptions()
    streaming_config.security_options.use_ephemeral_certs = True

    # Additional logging to verify streaming setup
    print(f"Attempting to start streaming with profile: {streaming_config.profile_name}")

    # Explicitly mention USB streaming here
    streaming_manager.streaming_config = streaming_config

    # Start streaming
    streaming_manager.start_streaming()

    # Setup subscription
    config = streaming_client.subscription_config
    config.subscriber_data_type = StreamingDataType.Rgb | StreamingDataType.Slam
    config.message_queue_size[StreamingDataType.Rgb] = 1
    config.message_queue_size[StreamingDataType.Slam] = 1

    streaming_client.subscribe()

    # Define observer class to process incoming images
    class StreamingClientObserver:
        def __init__(self):
            print("Observer initialized.")
            self.images = {}

        def on_image_received(self, image: np.array, record):
            print(f"Image received callback triggered for Camera ID: {record.camera_id}")
            if image is not None and record.camera_id == CameraId.Rgb:
                print(f"Received image shape: {image.shape}")
                # Convert image to BGR format for OpenCV if necessary
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                print("Processing image with YOLO...")
                results = model(image)  # Pass the image directly to YOLO
                annotated_frame = results[0].plot()

                # Display the YOLO-detected frame
                cv2.imshow("YOLO Live Detection", annotated_frame)

                # Handle quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()

    # Register the observer
    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)
    print("Subscribed to the stream.")

    # Keep the script running to process incoming frames
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stream interrupted by user.")
finally:
    # Cleanup
    if streaming_client:
        streaming_client.unsubscribe()
    if streaming_manager:
        try:
            streaming_manager.stop_streaming()
        except RuntimeError as e:
            print(f"Error during cleanup: {e}")
    if device_client and device:
        device_client.disconnect(device)

    print("Cleanup complete.")
