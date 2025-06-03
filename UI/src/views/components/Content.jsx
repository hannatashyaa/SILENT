import { useState, useRef } from "react";
import createContentPresenter from "../../presenters/ContentPresenter";
import cameraIcon from "../../assets/camera-icon.png";
import imageIcon from "../../assets/image-icon.png";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faCircleCheck } from "@fortawesome/free-regular-svg-icons";
import { faHands } from "@fortawesome/free-solid-svg-icons";
import { faHandPeace } from "@fortawesome/free-solid-svg-icons";
import { useNavigate } from "react-router-dom";

export default function Content() {
  const [showTranslate, setShowTranslate] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState("");
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isImageReady, setIsImageReady] = useState(false);
  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  const handleCameraClick = () => {
    navigate("/kamera");
  };

  const presenter = createContentPresenter({
    setShowTranslate,
    setSelectedLanguage,
    setPreviewUrl,
    setIsImageReady,
  });

  return (
    <section className="min-h-screen text-center" id="home">
      <section className="max-w-6xl mx-auto px-4 mt-12">
        <div className="flex items-center justify-center gap-4 md:gap-6">
          <div
            className="h-px bg-[#009DFF] flex-1 max-w-[200px]"
            style={{ boxShadow: "0px 3px 4px rgba(0, 0, 0, 0.3)" }}
          ></div>
          <h2
            className="text-center text-xl font-semibold text-black whitespace-nowrap"
            style={{ textShadow: "0px 3px 4px rgba(0, 0, 0, 0.3)" }}
          >
            SIBI & BISINDO
          </h2>
          <div
            className="h-px bg-[#009DFF] flex-1 max-w-[200px]"
            style={{ boxShadow: "0px 3px 4px rgba(0, 0, 0, 0.3)" }}
          ></div>
        </div>
        <p className="mb-8">
          Indonesia memiliki dua sistem bahasa isyarat utama yang digunakan oleh
          komunitas tuli.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
          {/* Kartu SIBI */}
          <div className="bg-[#EDF4FE] border border-gray-300 rounded-lg p-6 shadow-md flex flex-col items-start text-left transition-transform duration-300 transform hover:-translate-y-2 hover:shadow-lg">
            <div className="flex items-center gap-4 mb-4">
              <div className="bg-[#3D81F3] w-13 h-13 rounded-2xl flex items-center justify-center shadow-md">
                <FontAwesomeIcon
                  icon={faHandPeace}
                  size="lg"
                  style={{ color: "#ffffff" }}
                />
              </div>
              <div>
                <h3 className="text-lg font-semibold mb-1">SIBI</h3>
                <p className="text-sm text-gray-600">
                  Sistem Isyarat Bahasa Indonesia
                </p>
              </div>
            </div>
            <div className="w-full h-auto rounded-md mb-3 aspect-video">
              <iframe
                src="https://www.youtube.com/embed/cPGNqDzSBv4"
                title="YouTube video"
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
                className="w-full h-full rounded-md"
              ></iframe>
            </div>
            <p className="text-sm text-gray-700 text-left">
              SIBI adalah bahasa isyarat yang dikembangkan oleh pemerintah
              Indonesia. Bahasa isyarat SIBI rincang menyesuaikan struktur
              bahasa Indonesia. Pembuatan SIBI diadopsi dari bahasa isyarat
              Amerika (ASL).
            </p>
            <ul className="text-sm text-gray-700 text-left space-y-1">
              <li className="flex items-start gap-2">
                <FontAwesomeIcon
                  icon={faCircleCheck}
                  className="text-[#41CA9D] mt-1"
                />
                Mengikuti struktur bahasa Indonesia
              </li>
              <li className="flex items-start gap-2">
                <FontAwesomeIcon
                  icon={faCircleCheck}
                  className="text-[#41CA9D] mt-1"
                />
                Cocok untuk pembelajaran formal
              </li>
              <li className="flex items-start gap-2">
                <FontAwesomeIcon
                  icon={faCircleCheck}
                  className="text-[#41CA9D] mt-1"
                />
                Digunakan dalam konteks resmi
              </li>
            </ul>
            <div className="w-full flex justify-end mt-4">
              <button
                className="bg-[#3D81F3] text-white px-4 py-2 text-sm hover:bg-blue-700 transition"
                onClick={() => presenter.handleStart("SIBI")}
              >
                Mulai
              </button>
            </div>
          </div>

          {/* Kartu BISINDO */}
          <div className="bg-[#EEFDF8] border border-gray-300 rounded-lg p-6 shadow-md flex flex-col items-start text-left transition-transform duration-300 transform hover:-translate-y-2 hover:shadow-lg">
            <div className="flex items-center gap-4 mb-4">
              <div className="bg-[#36D298] w-16 h-16 rounded-2xl flex items-center justify-center shadow-md">
                <FontAwesomeIcon
                  icon={faHands}
                  size="lg"
                  style={{ color: "#ffffff" }}
                />
              </div>
              <div>
                <h3 className="text-lg font-semibold mb-1">BISINDO</h3>
                <p className="text-sm text-gray-600">
                  Bahasa Isyarat Indonesia
                </p>
              </div>
            </div>
            <div className="w-full h-auto rounded-md mb-3 aspect-video">
              <iframe
                src="https://www.youtube.com/embed/91kmtjMvOR4"
                title="YouTube video"
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
                className="w-full h-full rounded-md"
              ></iframe>
            </div>
            <p className="text-sm text-gray-700 text-left">
              Bisindo adalah Bahasa Isyarat Indonesia yang berlaku di Indonesia.
              BISINDO merupakan bahasa ibu yang tumbuh secara alami pada
              kalangan komunitas Tuli di Indonesia.
            </p>
            <ul className="text-sm text-gray-700 text-left space-y-1">
              <li className="flex items-start gap-2">
                <FontAwesomeIcon
                  icon={faCircleCheck}
                  className="text-[#41CA9D] mt-1"
                />
                Bahasa alami komunitas tuli.
              </li>
              <li className="flex items-start gap-2">
                <FontAwesomeIcon
                  icon={faCircleCheck}
                  className="text-[#41CA9D] mt-1"
                />
                Tata bahasa yang kaya visual.
              </li>
              <li className="flex items-start gap-2">
                <FontAwesomeIcon
                  icon={faCircleCheck}
                  className="text-[#41CA9D] mt-1"
                />
                Digunakan dalam kehidupan sehari-hari
              </li>
            </ul>
            <div className="w-full flex justify-end mt-4">
              <button
                className="bg-[#36D298] text-white px-4 py-2 text-sm hover:bg-[#2FB886] transition"
                onClick={() => presenter.handleStart("BISINDO")}
              >
                Mulai
              </button>
            </div>
          </div>
        </div>
      </section>

      {showTranslate && (
        <section className="max-w-6xl mx-auto px-4 py-20" id="translate">
          <div className="flex items-center justify-center gap-4 md:gap-6">
            <div
              className="h-px bg-[#009DFF] flex-1 max-w-[200px]"
              style={{ boxShadow: "0px 3px 4px rgba(0, 0, 0, 0.3)" }}
            ></div>
            <h2
              className="text-center text-xl font-semibold text-black whitespace-nowrap"
              style={{ textShadow: "0px 3px 4px rgba(0, 0, 0, 0.3)" }}
            >
              Metode Terjemahan
            </h2>
            <div
              className="h-px bg-[#009DFF] flex-1 max-w-[200px]"
              style={{ boxShadow: "0px 3px 4px rgba(0, 0, 0, 0.3)" }}
            ></div>
          </div>
          <p className="mb-8">
            Terjemahkan bahasa isyarat{" "}
            <span className="font-semibold">{selectedLanguage}</span> dengan
            mudah melalui berbagai metode di bawah.
          </p>
          <div className="flex justify-center gap-8">
            <div
              onClick={handleCameraClick}
              className="border p-4 rounded-lg shadow-md flex flex-col items-center justify-center text-center transition-transform duration-300 transform hover:-translate-y-2 hover:shadow-lg"
            >
              <img src={cameraIcon} alt="Camera" className="w-11 h-11 mb-3" />
              <h3 className="font-semibold text-lg mb-1">Kamera</h3>
              <p className="text-sm text-gray-500 leading-snug">
                Terjemahkan Bahasa Isyarat secara Real-time
              </p>
            </div>
            <div
              className="w-84 border p-4 rounded-lg shadow-md flex flex-col items-center justify-center text-center transition-transform duration-300 transform hover:-translate-y-2 hover:shadow-lg"
              onClick={() => fileInputRef.current?.click()}
            >
              <img src={imageIcon} alt="Image" className="w-11 h-11 mb-3" />
              <h3 className="font-semibold">Gambar</h3>
              <p className="text-sm text-gray-500 leading-snug">
                Unggah foto bahasa isyarat
              </p>
              <input
                type="file"
                accept="image/*"
                ref={fileInputRef}
                style={{ display: "none" }}
                onChange={presenter.handleImageUpload}
              />
            </div>
          </div>

          {isImageReady && previewUrl && (
            <div className="mt-6 flex flex-col items-center gap-4">
              <img
                src={previewUrl}
                alt="Preview"
                className="w-64 h-auto rounded-md border shadow-md"
              />
              <button
                onClick={() => alert("Proses terjemahan dimulai...")}
                className="bg-[#3D81F3] text-white px-4 py-2 rounded-md hover:bg-blue-700 transition"
              >
                Terjemahkan Sekarang
              </button>
            </div>
          )}
        </section>
      )}

      <section className="bg-[#D9D9D9] mt-10 py-10">
        <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-8 px-4">
          <div className="bg-white border border-gray-300 rounded-lg p-8 shadow">
            <h3 className="text-lg font-semibold mb-4">Translation</h3>
            <div className="flex flex-col items-center justify-center h-64 text-gray-500">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-12 w-12 mb-2"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M8 10h.01M12 10h.01M16 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <p className="text-center">
                Hasil terjemahan akan ditampilkan di sini ...
              </p>
            </div>
          </div>

          <div
            className="bg-white border border-gray-300 rounded-lg p-8 shadow"
            id="history"
          >
            <h3 className="text-lg font-semibold mb-4">History</h3>
            <div className="space-y-3">
              <div className="flex items-center p-3 bg-gray-100 rounded-lg">
                <img
                  src="https://via.placeholder.com/50"
                  alt="Sign Image"
                  className="w-12 h-12 object-contain mr-4"
                />
                <div>
                  <p className="font-medium">Terjemahan</p>
                  <span className="text-sm text-gray-500">B</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </section>
  );
}