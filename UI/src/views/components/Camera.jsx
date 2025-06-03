import React, { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faVideo } from "@fortawesome/free-solid-svg-icons";

const Camera = ({ active }) => {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [isPaused, setIsPaused] = useState(false);
  const navigate = useNavigate();
  const [isActive, setIsActive] = useState(false);

  useEffect(() => {
    const startCamera = async () => {
      try {
        const mediaStream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
        });
        streamRef.current = mediaStream;
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
        }
      } catch (err) {
        console.error("Error accessing camera: ", err);
      }
    };

    const stopCamera = () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    };

    if (isActive) {
      startCamera();
    } else {
      stopCamera();
    }

    return () => stopCamera(); // matikan saat unmount juga
  }, [isActive]);

  const handlePause = () => {
    const video = videoRef.current;
    if (!video) return;

    if (isPaused) {
      video.play();
    } else {
      video.pause();
    }
    setIsPaused(!isPaused);
  };

  const handleBack = () => {
    // Matikan kamera sebelum pindah halaman
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    navigate(-1);
  };

  return (
    <div className="w-full h-screen bg-[#D9D9D9] px-6 py-4 flex flex-col">
      {/* Tombol Back */}
      <button
        onClick={handleBack}
        className="absolute top-4 left-4 text-blue-500 font-semibold text-xl z-10"
      >
        back
      </button>

      {active && (
        <div className="flex flex-col md:flex-row justify-center items-center gap-6 flex-grow">
          {/* Video Preview */}
          <div className="flex flex-col items-center">
            <div className="w-full max-w-[740px] aspect-video bg-black rounded-xl overflow-hidden shadow">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                className="w-full h-full object-cover"
              />
            </div>

            {/* Teks Terjemahan di bawah video */}
            <div className="mt-4 bg-white text-blue-600 text-xl px-4 py-2 rounded shadow max-w-[740px] text-center">
              Terjemahan akan ditampilkan dibagian ini . . . !
            </div>
          </div>

          {/* Tombol Kontrol */}
          <div className="flex flex-row md:flex-col gap-5 items-center mt-4 md:mt-0">
            <button
              onClick={() => setIsActive((prev) => !prev)}
              className={`w-14 h-14 ${
                isActive ? "bg-red-500" : "bg-white"
              } rounded-full flex items-center justify-center shadow-md`}
              title={isActive ? "Stop Kamera" : "Mulai Kamera"}
            >
              {isActive ? (
                <div className="w-3.5 h-3.5 bg-white"></div> // tampilan kotak saat kamera aktif
              ) : (
                <FontAwesomeIcon icon={faVideo} style={{ color: "#000000" }} /> // ikon video saat belum aktif
              )}
            </button>

            <button
              onClick={handlePause}
              className="w-14 h-14 bg-black rounded-full flex items-center justify-center shadow-md"
              title={isPaused ? "Lanjutkan" : "Pause"}
            >
              <div
                className={`w-3.5 h-3.5 bg-white ${
                  isPaused ? "rounded" : "rounded-full"
                }`}
              ></div>
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Camera;
