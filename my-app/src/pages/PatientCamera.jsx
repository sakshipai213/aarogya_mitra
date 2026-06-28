import React, { useEffect, useRef, useState } from "react";

function PatientCamera() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [capturedImage, setCapturedImage] = useState(null); // dataURL string
  const [selectedFile, setSelectedFile] = useState(null);   // File from <input>
  const [generatedVideoUrl, setGeneratedVideoUrl] = useState(null);

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Error accessing camera:", err);
        alert("Unable to access camera. Please check permissions.");
      }
    };

    startCamera();
  }, []);

  const captureImage = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (video && canvas) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const dataUrl = canvas.toDataURL("image/png");
      setCapturedImage(dataUrl);
      setSelectedFile(null); // Clear file upload
      setGeneratedVideoUrl(null);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setCapturedImage(URL.createObjectURL(file)); // Preview
      setGeneratedVideoUrl(null);
    }
  };

  const handleGenerateVideo = async () => {
    let fileToSend;

    if (selectedFile) {
      fileToSend = selectedFile;
    } else if (capturedImage) {
      const blob = await (await fetch(capturedImage)).blob();
      fileToSend = new File([blob], "captured.png", { type: "image/png" });
    } else {
      alert("No image selected or captured!");
      return;
    }

    const formData = new FormData();
    formData.append("image", fileToSend);

    try {
      const response = await fetch("http://127.0.0.1:5000/generate-video", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const videoBlob = await response.blob();
        const videoURL = URL.createObjectURL(videoBlob);
        setGeneratedVideoUrl(videoURL);
        alert("✅ Video generated successfully!");
      } else {
        const error = await response.json();
        console.error("Video generation failed:", error);
        alert("⚠️ Video generation failed");
      }
    } catch (error) {
      console.error("Error generating video:", error);
      alert("❌ Something went wrong");
    }
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>📷 Upload or Capture Prescription</h2>

      <div style={styles.videoWrapper}>
        <video ref={videoRef} autoPlay playsInline style={styles.video} />
      </div>

      <button style={styles.captureButton} onClick={captureImage}>
        📸 Take Photo
      </button>

      <div style={{ margin: "1.5rem 0" }}>
        <input type="file" accept="image/*" onChange={handleFileChange} />
      </div>

      {(capturedImage || selectedFile) && (
        <div style={styles.previewWrapper}>
          <img src={capturedImage} alt="Selected" style={styles.previewImage} />
          <button style={styles.uploadButton} onClick={handleGenerateVideo}>
            🧠 Generate Video Instructions
          </button>
        </div>
      )}

      {generatedVideoUrl && (
        <div style={styles.videoPreviewWrapper}>
          <h3>🎥 Video Instructions</h3>
          <video controls width="100%" style={styles.videoPreview}>
            <source src={generatedVideoUrl} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        </div>
      )}

      <canvas ref={canvasRef} style={{ display: "none" }} />
    </div>
  );
}

const styles = {
  container: {
    maxWidth: "100%",
    padding: "1rem",
    backgroundColor: "#ffffff",
    fontFamily: "sans-serif",
    textAlign: "center",
  },
  title: {
    fontSize: "1.8rem",
    marginBottom: "1rem",
    fontWeight: "600",
  },
  videoWrapper: {
    display: "flex",
    justifyContent: "center",
  },
  video: {
    width: "90vw",
    maxWidth: "500px",
    borderRadius: "12px",
    objectFit: "cover",
    aspectRatio: "4 / 3",
    boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)",
  },
  captureButton: {
    marginTop: "1rem",
    fontSize: "1.1rem",
    padding: "0.6rem 1.4rem",
    borderRadius: "8px",
    backgroundColor: "#007BFF",
    color: "#fff",
    border: "none",
    cursor: "pointer",
  },
  previewWrapper: {
    marginTop: "1.5rem",
  },
  previewImage: {
    width: "90vw",
    maxWidth: "500px",
    borderRadius: "12px",
    marginBottom: "1rem",
  },
  uploadButton: {
    fontSize: "1.1rem",
    padding: "0.6rem 1.4rem",
    borderRadius: "8px",
    backgroundColor: "#28a745",
    color: "#fff",
    border: "none",
    cursor: "pointer",
  },
  videoPreviewWrapper: {
    marginTop: "2rem",
  },
  videoPreview: {
    width: "90vw",
    maxWidth: "500px",
    borderRadius: "12px",
  },
};

export default PatientCamera;
