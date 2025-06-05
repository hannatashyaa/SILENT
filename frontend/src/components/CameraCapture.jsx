import React, { useRef, useState, useEffect, useCallback } from 'react'
import { Camera, CameraOff, Loader, Square, Timer, Play, Pause, FlipHorizontal2, AlertCircle } from 'lucide-react'
import { apiService } from '../services/apiService'

const CameraCapture = ({ language, onPrediction }) => {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const timerRef = useRef(null)
  const autoIntervalRef = useRef(null)
  
  const [isStreaming, setIsStreaming] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [isCapturing, setIsCapturing] = useState(false)
  const [error, setError] = useState(null)
  const [lastCapture, setLastCapture] = useState(null)
  const [backendStatus, setBackendStatus] = useState(null)
  
  // Camera features
  const [isMirrored, setIsMirrored] = useState(true)
  const [isAutoCapture, setIsAutoCapture] = useState(false)
  const [timerMode, setTimerMode] = useState(false)
  const [timerSeconds, setTimerSeconds] = useState(3)
  const [countdown, setCountdown] = useState(0)
  const [captureCount, setCaptureCount] = useState(0)

  // Check backend connectivity on component mount
  useEffect(() => {
    checkBackendConnectivity()
  }, [])

  const checkBackendConnectivity = async () => {
    try {
      console.log('Checking backend connectivity...')
      const available = await apiService.isBackendAvailable()
      
      if (available) {
        console.log('Backend is available')
        setBackendStatus('connected')
        // Get additional info
        try {
          const status = await apiService.getApiStatus()
          console.log('Backend status:', status)
        } catch (err) {
          console.warn(' Could not get full backend status:', err)
        }
      } else {
        console.log(' Backend is not available')
        setBackendStatus('disconnected')
        setError('Backend server tidak tersedia. Pastikan server Python berjalan di http://localhost:5000')
      }
    } catch (err) {
      console.error(' Backend connectivity check failed:', err)
      setBackendStatus('error')
      setError(`Backend error: ${err.message}`)
    }
  }

  // ROBUST CAMERA START
  const startCamera = async () => {
    setError(null)
    setIsLoading(true)

    try {
      console.log('🎥 Requesting camera access...')
      
      if (!videoRef.current) {
        throw new Error('Video element not found')
      }
      
      const constraints = { 
        video: {
          facingMode: 'user',
          width: { ideal: 640 },
          height: { ideal: 480 }
        }
      }
      
      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      
      if (!stream || !stream.active) {
        throw new Error('Failed to get active camera stream')
      }
      
      streamRef.current = stream
      videoRef.current.srcObject = stream
      videoRef.current.autoplay = true
      videoRef.current.playsInline = true
      videoRef.current.muted = true
      
      await new Promise((resolve, reject) => {
        const video = videoRef.current
        
        const onLoadedMetadata = () => {
          console.log('📹 Video ready:', {
            width: video.videoWidth,
            height: video.videoHeight
          })
          cleanup()
          resolve()
        }
        
        const onError = (error) => {
          cleanup()
          reject(new Error('Video failed to load'))
        }
        
        const cleanup = () => {
          video.removeEventListener('loadedmetadata', onLoadedMetadata)
          video.removeEventListener('error', onError)
        }
        
        video.addEventListener('loadedmetadata', onLoadedMetadata)
        video.addEventListener('error', onError)
        
        setTimeout(() => {
          cleanup()
          reject(new Error('Video loading timeout'))
        }, 5000)
        
        if (video.readyState >= 1) {
          onLoadedMetadata()
        }
      })
      
      try {
        await videoRef.current.play()
      } catch (playError) {
        console.warn('Video play promise rejected (usually safe):', playError)
      }
      
      if (!videoRef.current.videoWidth || !videoRef.current.videoHeight) {
        throw new Error('Video dimensions not available')
      }
      
      setIsStreaming(true)
      setIsLoading(false)
      
      console.log('🎉 Camera ready!')
      
    } catch (err) {
      console.error('❌ Camera startup failed:', err)
      
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
        streamRef.current = null
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null
      }
      
      let errorMessage = 'Camera failed to start'
      
      if (err.name === 'NotAllowedError') {
        errorMessage = 'Camera permission denied - please allow camera access'
      } else if (err.name === 'NotFoundError') {
        errorMessage = 'No camera found - check camera connection'
      } else if (err.name === 'NotReadableError') {
        errorMessage = 'Camera is busy - close other apps using camera'
      } else if (err.name === 'OverconstrainedError') {
        errorMessage = 'Camera constraints not supported'
      } else if (err.message) {
        errorMessage = err.message
      }
      
      setError(errorMessage)
      setIsLoading(false)
      setIsStreaming(false)
    }
  }

  const stopCamera = () => {
    console.log('🛑 Stopping camera...')
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null
      videoRef.current.load()
    }
    
    setIsStreaming(false)
    setIsLoading(false)
    setIsCapturing(false)
    setIsAutoCapture(false)
    setError(null)
    
    clearTimers()
  }

  const clearTimers = () => {
    if (timerRef.current) clearTimeout(timerRef.current)
    if (autoIntervalRef.current) clearInterval(autoIntervalRef.current)
    setCountdown(0)
  }

  // Capture image from video stream
  const captureImage = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || !isStreaming) return null

    const video = videoRef.current
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    if (isMirrored) {
      ctx.scale(-1, 1)
      ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height)
      ctx.scale(-1, 1)
    } else {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    }

    return new Promise((resolve) => {
      canvas.toBlob(resolve, 'image/jpeg', 0.8)
    })
  }, [isStreaming, isMirrored])

  // FIXED: Predict from camera capture
  const predictFromCamera = useCallback(async () => {
    if (!isStreaming || !videoRef.current) return

    // Check backend status first
    if (backendStatus !== 'connected') {
      setError('Backend tidak tersedia. Cek koneksi ke server Python.')
      return
    }

    try {
      setIsCapturing(true)
      setError(null)

      console.log('Capturing image from camera...')
      
      const imageBlob = await captureImage()
      if (!imageBlob) {
        throw new Error('Failed to capture image from camera')
      }

      console.log('Image captured, making prediction...')
      console.log('Using language:', language)

      setCaptureCount(prev => prev + 1)

      // Save last capture for preview
      const imageUrl = URL.createObjectURL(imageBlob)
      if (lastCapture) URL.revokeObjectURL(lastCapture)
      setLastCapture(imageUrl)

      // FIXED: Use the corrected API service
      console.log('Sending to backend API...')
      const result = await apiService.predictImage(imageBlob, language)
      
      console.log('Prediction result:', result)
      
      // Send result to parent component
      onPrediction(result, imageUrl)

    } catch (err) {
      console.error(' Prediction error:', err)
      
      let errorMessage = err.message || 'Failed to predict image'
      
      // Handle specific error types
      if (err.message.includes('Network error')) {
        errorMessage = 'Tidak bisa menghubungi server. Pastikan backend Python berjalan di http://localhost:5000'
        setBackendStatus('disconnected')
      } else if (err.message.includes('timeout')) {
        errorMessage = 'Server terlalu lama merespons. Coba lagi.'
      }
      
      setError(errorMessage)
      
      onPrediction({
        success: false,
        error: errorMessage
      })
    } finally {
      setIsCapturing(false)
    }
  }, [isStreaming, captureImage, language, onPrediction, lastCapture, backendStatus])

  // Timer capture
  const startTimerCapture = () => {
    if (!isStreaming) return

    setCountdown(timerSeconds)
    
    const timer = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          clearInterval(timer)
          predictFromCamera()
          return 0
        }
        return prev - 1
      })
    }, 1000)
    
    timerRef.current = timer
  }

  // Auto capture mode
  const toggleAutoCapture = () => {
    if (isAutoCapture) {
      if (autoIntervalRef.current) {
        clearInterval(autoIntervalRef.current)
        autoIntervalRef.current = null
      }
      setIsAutoCapture(false)
    } else {
      setIsAutoCapture(true)
      autoIntervalRef.current = setInterval(() => {
        if (!isCapturing && isStreaming && backendStatus === 'connected') {
          predictFromCamera()
        }
      }, 3000)
    }
  }

  const toggleMirror = () => {
    setIsMirrored(!isMirrored)
  }

  // Auto-start camera when component mounts
  useEffect(() => {
    console.log('🚀 Component mounted - checking video element...')
    
    const checkAndStart = () => {
      if (videoRef.current) {
        console.log('Video element found, starting camera...')
        startCamera()
      } else {
        console.log('Video element not ready, retrying...')
        setTimeout(checkAndStart, 100)
      }
    }
    
    const initTimer = setTimeout(checkAndStart, 200)
    
    return () => {
      clearTimeout(initTimer)
      stopCamera()
      clearTimers()
      if (lastCapture) {
        URL.revokeObjectURL(lastCapture)
      }
    }
  }, [])

  return (
    <div className="space-y-4">
      {/* Backend Status Indicator */}
      {backendStatus && (
        <div className="flex items-center justify-between p-3 rounded-lg border">
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${
              backendStatus === 'connected' ? 'bg-green-500' : 
              backendStatus === 'disconnected' ? 'bg-red-500' : 
              'bg-yellow-500'
            }`}></div>
            <span className="text-sm font-medium">
             {
                backendStatus === 'connected' ? 'Connected' :
                backendStatus === 'disconnected' ? 'Disconnected' :
                'Checking...'
              }
            </span>
          </div>
          
          {backendStatus !== 'connected' && (
            <button
              onClick={checkBackendConnectivity}
              className="text-blue-500 hover:text-blue-700 text-sm"
            >
              Retry Connection
            </button>
          )}
        </div>
      )}

      {/* Camera Controls Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Camera className="w-5 h-5 text-blue-600" />
          Camera Capture
          {isStreaming && (
            <span className="bg-green-100 text-green-800 text-xs px-2 py-1 rounded-full">
              Live
            </span>
          )}
        </h3>
        
        <div className="flex items-center gap-2">
          {captureCount > 0 && (
            <span className="text-sm text-gray-600 bg-gray-100 px-2 py-1 rounded">
              📸 {captureCount}
            </span>
          )}
          
          {!isStreaming && !isLoading ? (
            <button
              onClick={startCamera}
              className="btn-primary flex items-center gap-2"
            >
              <Camera className="w-4 h-4" />
              Start Camera
            </button>
          ) : isLoading ? (
            <button
              disabled
              className="bg-gray-400 text-white px-4 py-2 rounded-lg flex items-center gap-2 cursor-not-allowed"
            >
              <Loader className="w-4 h-4 animate-spin" />
              Starting...
            </button>
          ) : (
            <button
              onClick={stopCamera}
              className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg flex items-center gap-2"
            >
              <CameraOff className="w-4 h-4" />
              Stop Camera
            </button>
          )}
        </div>
      </div>

      {/* Backend Connection Error */}
      {/* {backendStatus === 'disconnected' && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <AlertCircle className="w-5 h-5 text-red-500" />
            <h4 className="font-medium text-red-800">Backend Tidak Tersedia</h4>
          </div>
          <p className="text-red-700 text-sm mb-3">
            Server Python backend tidak dapat dihubungi. Pastikan:
          </p>
          <ul className="text-red-600 text-sm space-y-1 mb-3">
            <li>• Server Python berjalan di <code>http://localhost:5000</code></li>
            <li>• Jalankan <code>python app.py</code> di folder backend</li>
            <li>• Model telah ditraining dan tersedia</li>
            <li>• Tidak ada firewall yang memblokir koneksi</li>
          </ul>
          <button
            onClick={checkBackendConnectivity}
            className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded text-sm"
          >
            Test Connection Lagi
          </button>
        </div>
      )} */}

      {/* Camera Preview */}
      <div className="relative">
        <div className="camera-preview bg-gray-100 rounded-lg overflow-hidden" style={{ aspectRatio: '5/3' }}>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className={`w-full h-full object-cover ${isMirrored ? 'scale-x-[-1]' : ''} ${
              isStreaming ? 'block' : 'hidden'
            }`}
          />
          
          {isLoading && (
            <div className="absolute inset-0 w-full h-full flex items-center justify-center bg-blue-50">
              <div className="text-center">
                <Camera className="w-12 h-12 text-blue-500 mx-auto mb-2 animate-pulse" />
                <p className="text-blue-700 font-medium">Connecting to camera...</p>
                <div className="mt-2">
                  <div className="inline-block animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                </div>
              </div>
            </div>
          )}
          
          {!isLoading && !isStreaming && (
            <div className="absolute inset-0 w-full h-full flex items-center justify-center bg-gray-100">
              <div className="text-center">
                <Camera className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600 font-medium mb-2">Camera Auto-Start</p>
                <button 
                  onClick={startCamera}
                  disabled={isLoading}
                  className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-3 rounded-lg font-medium flex items-center gap-2 mx-auto disabled:opacity-50"
                >
                  <Camera className="w-4 h-4" />
                  {isLoading ? 'Starting...' : 'Manual Start'}
                </button>
              </div>
            </div>
          )}
        </div>

        {isCapturing && (
          <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center rounded-lg">
            <div className="text-center text-white">
              <Loader className="w-8 h-8 animate-spin mx-auto mb-2" />
              <p>Predicting...</p>
            </div>
          </div>
        )}

        {isAutoCapture && (
          <div className="absolute top-4 left-4 bg-green-500 text-white px-3 py-1 rounded-full text-sm flex items-center gap-1">
            <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
            Auto Mode
          </div>
        )}

        {countdown > 0 && (
          <div className="absolute inset-0 bg-black bg-opacity-70 flex items-center justify-center rounded-lg">
            <div className="text-center text-white">
              <div className="text-6xl font-bold mb-2">{countdown}</div>
              <p className="text-lg">Get ready...</p>
            </div>
          </div>
        )}

        {isMirrored && isStreaming && (
          <div className="absolute top-4 right-4 bg-blue-500 text-white px-2 py-1 rounded text-xs">
            Mirror
          </div>
        )}
      </div>

      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {/* Camera Features - Only show if camera is ready and backend is connected */}
      {isStreaming && backendStatus === 'connected' && (
        <div className="space-y-4">
          {/* Mirror & Settings */}
          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <span className="text-sm font-medium text-gray-700">Camera Settings</span>
            <div className="flex items-center gap-2">
              <button
                onClick={toggleMirror}
                className={`p-2 rounded-lg flex items-center gap-1 text-sm ${
                  isMirrored 
                    ? 'bg-blue-500 text-white' 
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
                title="Toggle Mirror Mode"
              >
                <FlipHorizontal2 className="w-4 h-4" />
                Mirror
              </button>
            </div>
          </div>

          {/* Capture Controls */}
          <div className="grid grid-cols-2 gap-3">
            <button
              onClick={predictFromCamera}
              disabled={!isStreaming || isCapturing || backendStatus !== 'connected'}
              className="btn-primary flex items-center justify-center gap-2 disabled:opacity-50"
            >
              {isCapturing ? (
                <Loader className="w-4 h-4 animate-spin" />
              ) : (
                <Square className="w-4 h-4" />
              )}
              {isCapturing ? 'Predicting...' : 'Capture Now'}
            </button>

            <button
              onClick={startTimerCapture}
              disabled={!isStreaming || isCapturing || countdown > 0 || backendStatus !== 'connected'}
              className="btn-secondary flex items-center justify-center gap-2 disabled:opacity-50"
            >
              <Timer className="w-4 h-4" />
              Timer ({timerSeconds}s)
            </button>

            <button
              onClick={toggleAutoCapture}
              disabled={backendStatus !== 'connected'}
              className={`flex items-center justify-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors disabled:opacity-50 ${
                isAutoCapture
                  ? 'bg-green-500 hover:bg-green-600 text-white'
                  : 'bg-gray-200 hover:bg-gray-300 text-gray-700'
              }`}
            >
              {isAutoCapture ? (
                <>
                  <Pause className="w-4 h-4" />
                  Stop Auto
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Auto Capture
                </>
              )}
            </button>

            <div className="flex items-center gap-1">
              <select
                value={timerSeconds}
                onChange={(e) => setTimerSeconds(Number(e.target.value))}
                className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm"
              >
                <option value={3}>3 seconds</option>
                <option value={5}>5 seconds</option>
                <option value={10}>10 seconds</option>
              </select>
            </div>
          </div>
        </div>
      )}

      {/* Last Capture Preview */}
      {lastCapture && (
        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="text-sm font-medium text-gray-700 mb-2">Last Capture:</h4>
          <img
            src={lastCapture}
            alt="Last capture"
            className="w-32 h-24 object-cover rounded-lg border border-gray-200 mx-auto"
          />
        </div>
      )}

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h4 className="font-medium text-blue-900 mb-2">📋 Petunjuk Penggunaan:</h4>
        <ul className="text-blue-800 text-sm space-y-1">
          {/* <li>• <strong>Backend Required:</strong> Pastikan server Python berjalan di localhost:5000</li> */}
          <li>• <strong>Capture Now:</strong> Ambil foto langsung untuk prediksi</li>
          <li>• <strong>Timer:</strong> Ambil foto dengan hitungan mundur</li>
          <li>• <strong>Auto Capture:</strong> Ambil foto otomatis setiap 3 detik</li>
          <li>• <strong>Mirror:</strong> Balik tampilan kamera (seperti cermin)</li>
          <li>• Pastikan tangan berada dalam frame dengan pencahayaan cukup</li>
        </ul>
      </div>
    </div>
  )
}

export default CameraCapture