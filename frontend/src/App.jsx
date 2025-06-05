import React, { useState, createContext, useContext } from 'react';
import { Routes, Route, Navigate, Link } from 'react-router-dom';
import Header from './components/Header';
import Footer from './components/Footer';
import LanguageSelector from './components/LanguageSelector';
import CameraCapture from './components/CameraCapture';
import ImageUpload from './components/ImageUpload';
import PredictionResult from './components/PredictionResult';
import ModelInfo from './components/ModelInfo';
import landingPage from './assets/landingPage.png';
import cameraIcon from './assets/camera-icon.png' // atau sesuai path kamu
import imageIcon from './assets/image-icon.png'
import klik from './assets/klik.png'
import sibiImage from './assets/sibiringkas.jpg'
import bisindoImage from './assets/bisindoringkas.jpg'

// Create Global History Context
const HistoryContext = createContext();

// History Provider Component
const HistoryProvider = ({ children }) => {
  const [globalHistory, setGlobalHistory] = useState([]);

  const addToHistory = (result, capturedImageUrl = null, language = null) => {
    if (result && result.success) {
      const newEntry = {
        id: Date.now(),
        prediction: result.prediction,
        confidence: result.confidence,
        timestamp: new Date(),
        imageUrl: capturedImageUrl,
        language: result.dataset || language?.toUpperCase() || 'UNKNOWN',
        source: 'camera' // or 'upload'
      };
      setGlobalHistory(prev => [newEntry, ...prev.slice(0, 49)]); // Keep last 50 entries
    }
  };

  const clearHistory = () => {
    setGlobalHistory([]);
  };

  return (
    <HistoryContext.Provider value={{ 
      globalHistory, 
      addToHistory, 
      clearHistory 
    }}>
      {children}
    </HistoryContext.Provider>
  );
};

// Hook to use History Context
const useHistory = () => {
  const context = useContext(HistoryContext);
  if (!context) {
    throw new Error('useHistory must be used within HistoryProvider');
  }
  return context;
};

// --- Komponen YouTubeEmbed (Perbaikan URL Embed) ---
const YouTubeEmbed = ({ videoId, title, description }) => {
  return (
    <div className="youtube-card bg-white rounded-lg shadow-md overflow-hidden transform transition-transform duration-300 hover:scale-[1.02] cursor-pointer">
      <div className="relative" style={{ paddingBottom: '56.25%', height: 0 }}>
        {/* URL Embed YouTube yang benar */}
        <iframe
          src={`https://www.youtube.com/embed/${videoId}`}
          title={title}
          frameBorder="0"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
          className="absolute top-0 left-0 w-full h-full"
        ></iframe>
      </div>
      <div className="p-4">
        <h3 className="text-lg font-semibold text-gray-800 mb-1">{title}</h3>
        {description && <p className="text-gray-600 text-sm">{description}</p>}
      </div>
    </div>
  );
};

// Home Page Component
function HomePage() {
  const handleTryMeNowScroll = (event) => {
    event.preventDefault();
    const videoSection = document.getElementById('youtube-videos');
    if (videoSection) {
      videoSection.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const headerHeight = '64px';
  return (
    <div className="flex flex-col min-h-screen bg-[#009DFF]">
      <Header />

      <div
        className="bg-[#009DFF] flex items-center justify-center relative z-0"
        style={{ minHeight: `calc(100vh - ${headerHeight})` }}
      >
        <div className="container mx-auto px-4 py-16 flex flex-col md:flex-row items-center justify-between w-full">
          <div className="flex-1 max-w-lg text-center md:text-left mb-8 md:mb-0">
            <h1 className="text-5xl font-bold text-white mb-4">
              SILENT
            </h1>
            <p className="text-xl text-blue-100 mb-8">
              sign language interpretation<br />
              and expression translator
            </p>
            <a
              href="#youtube-videos"
              onClick={handleTryMeNowScroll}
              className="bg-blue-600 hover:bg-blue-900 text-white px-8 py-3 rounded-lg font-medium transition-all duration-200 transform hover:scale-105 flex items-center gap-2 w-fit mx-auto md:mx-0"
            >
              Pelajari Lebih Lanjut
              <span className="text-lg">👇</span>
            </a>
          </div>
          <div className="flex justify-center">
            <div className="relative">
              {/* <div className="w-80 h-80 bg-blue-600 rounded-full flex items-center justify-center"> */}
              <img src={landingPage} className="w-70 h-70 object-contain rounded-lg" alt="Landing Page Illustration" />
              {/* </div> */}
            </div>
          </div>
        </div>
      </div>

      {/* Bagian YouTube Videos */}
      <section id="youtube-videos" className="bg-gray-100 py-12">
        <div className="container mx-auto px-4">
          <h2 className="text-3xl font-bold text-center text-gray-800 mb-8">Pelajari Bahasa Isyarat</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <YouTubeEmbed
              videoId="NaafQwd0XEY"
              title="Bahasa Isyarat Indonesia (BISINDO)"
              description="Bisindo adalah Bahasa Isyarat Indonesia yang berlaku di Indonesia.
              BISINDO merupakan bahasa ibu yang tumbuh secara alami pada
              kalangan komunitas Tuli di Indonesia."
            />
            <YouTubeEmbed
              videoId="03kWuwWQwu0"
              title="Sistem Isyarat Bahasa Indonesia (SIBI)"
              description="SIBI adalah bahasa isyarat yang dikembangkan oleh pemerintah
              Indonesia. Bahasa isyarat SIBI rincang menyesuaikan struktur
              bahasa Indonesia. Pembuatan SIBI diadopsi dari bahasa isyarat
              Amerika (ASL)."
            />
          </div>
          {/* Gambar perbandingan SIBI & BISINDO */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-12">

            {/* Gambar BISINDO */}
            <div className="bg-white rounded-lg shadow-md p-4">
              <img
                src={bisindoImage}
                alt="Gambar ringkasan BISINDO"
                className="w-full max-w-[500px] h-auto mx-auto rounded-md"
              />
              <h3 className="text-lg font-semibold text-gray-800 text-center mt-4 mb-2">Bahasa Isyarat Indonesia (BISINDO)</h3>
              <p className="text-s text-gray-600 text-center">
               Bahasa isyarat alami komunitas tunarungu.
              </p>
            </div>

            {/* Gambar SIBI */}
            <div className="bg-white rounded-lg shadow-md p-4">
              <img
                src={sibiImage}
                alt="Gambar ringkasan SIBI"
                className="w-full max-w-[500px] h-auto mx-auto rounded-md"
              />
              <h3 className="text-lg font-semibold text-gray-800 text-center mt-4 mb-2">
                Sistem Isyarat Bahasa Indonesia (SIBI)
              </h3>
              <p className="text-sm text-gray-600 text-center">
               Sistem isyarat formal berdasarkan tata bahasa Indonesia.
              </p>
            </div>

          </div>
          <div className="text-center mt-12">
            <Link
              to="/translate"
              className="bg-blue-600 hover:bg-blue-700 text-white px-10 py-4 rounded-lg font-bold text-lg transition-all duration-200 transform hover:scale-105 inline-flex items-center gap-3"
            >
              Mulai Terjemahkan Sekarang
              <img src={klik} className="w-8 h-8 object-contain rounded-lg p-1" />
            </Link>
          </div>
        </div>
      </section>


      <Footer />
    </div>
  );
}

function TranslatePage() {
  const [selectedLanguage, setSelectedLanguage] = useState('bisindo');
  const [currentMode, setCurrentMode] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [localHistory, setLocalHistory] = useState([]);
  
  // Use global history context
  const { addToHistory } = useHistory();

  const handlePrediction = (result, capturedImageUrl = null) => {
    setPredictionResult(result);
    
    if (result && result.success) {
      // Add to global history
      addToHistory(result, capturedImageUrl, selectedLanguage);
      
      // Keep local history for sidebar display (last 10 items)
      const newEntry = {
        id: Date.now(),
        prediction: result.prediction,
        confidence: result.confidence,
        timestamp: new Date(),
        imageUrl: capturedImageUrl,
        language: result.dataset || selectedLanguage.toUpperCase()
      };
      setLocalHistory(prev => [newEntry, ...prev.slice(0, 9)]);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />

      <main className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <LanguageSelector
            selectedLanguage={selectedLanguage}
            onLanguageChange={setSelectedLanguage}
          />
        </div>

        <div className="mb-8">
          <h2 className="text-2xl font-semibold text-center mb-2">Metode Terjemahan</h2>
          <p className="text-gray-600 text-center mb-8">
            Terjemahkan bahasa isyarat BISINDO & SIBI dengan mudah<br />
            melalui berbagai metode di bawah.
          </p>

          <div className="grid md:grid-cols-2 gap-6 max-w-2xl mx-auto">
            <div
              className={`card-hover ${currentMode === 'camera' ? 'active' : ''}`}
              onClick={() => setCurrentMode('camera')}
            >
              <div className="text-center">
                <div className={`w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 ${currentMode === 'camera' ? 'bg-blue-500 text-white' : 'bg-purple-100'
                  }`}>
                  <img
                    src={cameraIcon}
                    alt="Camera Icon"
                    className={`w-8 h-8 ${currentMode === 'camera' ? 'invert brightness-0' : ''
                      }`}
                  />
                </div>
                <h3 className={`text-lg font-semibold mb-2 ${currentMode === 'camera' ? 'text-blue-600' : 'text-gray-800'
                  }`}>Kamera</h3>
                <p className="text-gray-600 text-sm">
                  Terjemahkan bahasa isyarat secara Real-time
                </p>
                {currentMode === 'camera' && (
                  <div className="mt-3 flex items-center justify-center">
                    <span className="text-blue-600 text-sm font-medium flex items-center gap-1">
                      <span className="w-2 h-2 bg-blue-600 rounded-full animate-pulse"></span>
                      Aktif
                    </span>
                  </div>
                )}
              </div>
            </div>

            <div
              className={`card-hover ${currentMode === 'upload' ? 'active' : ''}`}
              onClick={() => setCurrentMode('upload')}
            >
              <div className="text-center">
                <div className={`w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 ${currentMode === 'upload' ? 'bg-blue-500 text-white' : 'bg-pink-100'
                  }`}>
                  <img
                    src={imageIcon}
                    alt="Image Icon"
                    className={`w-8 h-8 ${currentMode === 'upload' ? 'invert brightness-0' : ''
                      }`}
                  />
                </div>
                <h3 className={`text-lg font-semibold mb-2 ${currentMode === 'upload' ? 'text-blue-600' : 'text-gray-800'
                  }`}>Gambar</h3>
                <p className="text-gray-600 text-sm">
                  Unggah multiple foto (additive mode)
                </p>
                {currentMode === 'upload' && (
                  <div className="mt-3 flex items-center justify-center">
                    <span className="text-blue-600 text-sm font-medium flex items-center gap-1">
                      <span className="w-2 h-2 bg-blue-600 rounded-full animate-pulse"></span>
                      Aktif
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2">
            <div className="card">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <span>Translation</span>
                {currentMode && (
                  <span className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full">
                    {currentMode === 'camera' ? '📷 Camera Mode' : '🖼️ Upload Mode'}
                  </span>
                )}
              </h3>

              {!currentMode && (
                <div className="text-center py-12">
                  <div className="text-6xl mb-4">💻</div>
                  <p className="text-gray-500">
                    Pilih metode terjemahan di atas untuk memulai<br />
                    Hasil akan ditampilkan di sini...
                  </p>
                </div>
              )}

              {currentMode === 'camera' && (
                <div className="fade-in">
                  <CameraCapture
                    language={selectedLanguage}
                    onPrediction={(result, imageUrl) => handlePrediction(result, imageUrl)}
                  />
                </div>
              )}

              {currentMode === 'upload' && (
                <ImageUpload
                  language={selectedLanguage}
                  onPrediction={(result, imageUrl) => handlePrediction(result, imageUrl)}
                />
              )}

              {predictionResult && (
                <div className="mt-6">
                  <PredictionResult result={predictionResult} />
                </div>
              )}
            </div>
          </div>

          <div className="lg:col-span-1">
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">Recent History</h3>
                <Link 
                  to="/history" 
                  className="text-blue-600 hover:text-blue-800 text-sm font-medium"
                >
                  View All →
                </Link>
              </div>

              {localHistory.length === 0 ? (
                <div className="text-center py-8">
                  <div className="text-4xl mb-2">📝</div>
                  <p className="text-gray-500 text-sm">
                    History terjemahan akan<br />
                    muncul di sini...
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  {localHistory.slice(0, 5).map((item) => (
                    <div key={item.id} className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                      <div className="w-12 h-12 bg-orange-100 rounded-lg flex items-center justify-center overflow-hidden flex-shrink-0">
                        {item.imageUrl ? (
                          <img
                            src={item.imageUrl}
                            alt={`Sign ${item.prediction}`}
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <span className="text-lg">👋</span>
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                          <div className="font-semibold text-gray-800">Terjemahan</div>
                          <span className="text-xs text-gray-500 bg-gray-200 px-2 py-1 rounded">
                            {item.language}
                          </span>
                        </div>
                        <div className="text-2xl font-bold text-blue-600">{item.prediction}</div>
                        <div className="flex items-center justify-between mt-1">
                          <span className="text-xs text-gray-500">
                            {item.timestamp.toLocaleTimeString([], {
                              hour: '2-digit',
                              minute: '2-digit'
                            })}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

      </main>

      <Footer />
    </div>
  );
}

// About Page Component (Tidak ada perubahan)
function AboutPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      <main className="container mx-auto px-4 py-16">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-4xl font-bold text-center mb-8">About SILENT</h1>

          <div className="grid md:grid-cols-2 gap-8">
            <div className="card">
              <h2 className="text-2xl font-semibold mb-4">Our Mission</h2>
              <p className="text-gray-600 mb-4">
                SILENT adalah aplikasi web inovatif yang dirancang untuk menerjemahkan bahasa isyarat Indonesia (BISINDO dan SIBI) menjadi teks menggunakan teknologi computer vision dan machine learning.
              </p>
              <p className="text-gray-600">
                Kami bertujuan untuk membantu komunikasi antara komunitas tuna rungu dan masyarakat umum melalui teknologi yang mudah diakses.
              </p>
            </div>

            <div className="card">
              <h2 className="text-2xl font-semibold mb-4">Features</h2>
              <ul className="space-y-8 text-gray-600">
                <li className="flex items-center gap-2">
                  <span className="text-green-500">✓</span>
                  Real-time Translation
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-green-500">✓</span>
                  Image Upload Support
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-green-500">✓</span>
                  BISINDO & SIBI Support
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-green-500">✓</span>
                  Translation History
                </li>
              </ul>
            </div>
          </div>

          <div className="card mt-8">
            <h2 className="text-2xl font-semibold mb-4">Development Team</h2>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div className="text-center">
                <h3 className="font-semibold">Hanna Tashya Portuna</h3>
                <p className="text-gray-600">ML Engineer</p>
                <p className="text-sm text-gray-500">MC185D5X0288</p>
              </div>
              <div className="text-center">
                <h3 className="font-semibold">Evan Austin</h3>
                <p className="text-gray-600">ML Engineer</p>
                <p className="text-sm text-gray-500">MC185D5Y0640</p>
              </div>
              <div className="text-center">
                <h3 className="font-semibold">Mark Dionisius Alvin M G</h3>
                <p className="text-gray-600">ML Engineer</p>
                <p className="text-sm text-gray-500">MC185D5Y1221</p>
              </div>
              <div className="text-center">
                <h3 className="font-semibold">Femilia Zahrotun Nisa</h3>
                <p className="text-gray-600">Frontend Developer</p>
                <p className="text-sm text-gray-500">FC525D5X0011</p>
              </div>
              <div className="text-center">
                <h3 className="font-semibold">Icha Prisylia Br Ginting</h3>
                <p className="text-gray-600">Backend Developer</p>
                <p className="text-sm text-gray-500">FC525D5X0148</p>
              </div>
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}

// Updated History Page Component
function HistoryPage() {
  const { globalHistory, clearHistory } = useHistory();

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      <main className="container mx-auto px-4 py-16">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center justify-between mb-8">
            <h1 className="text-4xl font-bold">Translation History</h1>
            {globalHistory.length > 0 && (
              <button
                onClick={clearHistory}
                className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
              >
                Clear All History
              </button>
            )}
          </div>

          <div className="card">
            {globalHistory.length === 0 ? (
              <div className="text-center py-16">
                <div className="text-6xl mb-4">📝</div>
                <h3 className="text-xl font-semibold text-gray-700 mb-2">No History Yet</h3>
                <p className="text-gray-500 mb-6">
                  Start translating sign language to see your history here
                </p>
                <Link
                  to="/translate"
                  className="btn-primary inline-flex items-center gap-2"
                >
                  Start Translating
                </Link>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="mb-4 p-3 bg-blue-50 rounded-lg">
                  <p className="text-blue-800 text-sm">
                    📊 Total Translations: <strong>{globalHistory.length}</strong>
                  </p>
                </div>
                
                {globalHistory.map((item, index) => (
                  <div key={item.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                    <div className="flex items-center gap-4">
                      <div className="w-16 h-16 bg-blue-100 rounded-lg flex items-center justify-center overflow-hidden flex-shrink-0">
                        {item.imageUrl ? (
                          <img
                            src={item.imageUrl}
                            alt={`Sign ${item.prediction}`}
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <span className="text-2xl font-bold text-blue-600">{item.prediction}</span>
                        )}
                      </div>
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          <h3 className="font-semibold text-lg">Letter {item.prediction}</h3>
                          <span className="bg-gray-200 text-gray-700 px-2 py-1 rounded text-xs">
                            #{index + 1}
                          </span>
                        </div>
                        <div className="flex items-center gap-4 text-sm text-gray-600">
                          <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded">
                            {item.language}
                          </span>
                          <span>📅 {item.timestamp.toLocaleDateString()}</span>
                          <span>🕐 {item.timestamp.toLocaleTimeString()}</span>
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-semibold text-green-600">
                        {(item.confidence * 100).toFixed(1)}%
                      </div>
                      <div className="text-sm text-gray-500">Confidence</div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}

// Main App Component with History Provider
function App() {
  return (
    <HistoryProvider>
      <div className="App">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/translate" element={<TranslatePage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </HistoryProvider>
  );
}

export default App;