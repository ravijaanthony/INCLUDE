import { useEffect, useState } from 'react';
import { VideoInput } from './components/VideoInput';
import { ProcessingPipeline } from './components/ProcessingPipeline';
import { KeypointsDisplay } from './components/KeypointsDisplay';
import { TranslationOutput } from './components/TranslationOutput';
import { Sidebar } from './components/Sidebar';

type PredictionResult = {
  label: string;
  score: number;
  elapsed_ms: number;
  uid?: string;
};

const RECORDING_MS = 2000;

export default function App() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!videoFile) {
      setPreviewUrl(null);
      return;
    }

    const url = URL.createObjectURL(videoFile);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [videoFile]);

  useEffect(() => {
    return () => {
      cameraStream?.getTracks().forEach((track) => track.stop());
    };
  }, [cameraStream]);

  const stopCamera = () => {
    cameraStream?.getTracks().forEach((track) => track.stop());
    setCameraStream(null);
  };

  const handleToggleCamera = async () => {
    if (cameraStream) {
      stopCamera();
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
      setCameraStream(stream);
      setVideoFile(null);
      setPrediction(null);
      setError(null);
      setCurrentStep(0);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unable to access camera.';
      setError(message);
    }
  };

  const handleVideoUpload = (file: File) => {
    if (cameraStream) {
      stopCamera();
    }
    setVideoFile(file);
    setPrediction(null);
    setError(null);
    setCurrentStep(0);
  };

  const recordClip = (stream: MediaStream, durationMs: number) => {
    return new Promise<Blob>((resolve, reject) => {
      const options: MediaRecorderOptions = {};
      if (MediaRecorder.isTypeSupported('video/webm;codecs=vp9')) {
        options.mimeType = 'video/webm;codecs=vp9';
      } else if (MediaRecorder.isTypeSupported('video/webm;codecs=vp8')) {
        options.mimeType = 'video/webm;codecs=vp8';
      } else if (MediaRecorder.isTypeSupported('video/webm')) {
        options.mimeType = 'video/webm';
      }

      const recorder = new MediaRecorder(stream, options);
      const chunks: BlobPart[] = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };
      recorder.onerror = () => reject(new Error('Recording failed.'));
      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: recorder.mimeType || 'video/webm' });
        resolve(blob);
      };

      recorder.start();
      window.setTimeout(() => {
        if (recorder.state !== 'inactive') {
          recorder.stop();
        }
      }, durationMs);
    });
  };

  const requestPrediction = async (blob: Blob, filename: string) => {
    const form = new FormData();
    form.append('file', blob, filename);
    const resp = await fetch('/predict', { method: 'POST', body: form });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data?.error || `Request failed (${resp.status})`);
    }
    return data as PredictionResult;
  };

  const handleStartTranslation = async () => {
    if (isProcessing) return;
    if (!videoFile && !cameraStream) {
      setError('Please upload a video or enable the camera.');
      return;
    }

    setIsProcessing(true);
    setCurrentStep(1);
    setPrediction(null);
    setError(null);

    try {
      let blob: Blob;
      let filename: string;

      if (videoFile) {
        blob = videoFile;
        filename = videoFile.name;
      } else if (cameraStream) {
        setCurrentStep(2);
        blob = await recordClip(cameraStream, RECORDING_MS);
        filename = 'recording.webm';
      } else {
        setIsProcessing(false);
        return;
      }

      setCurrentStep(3);
      const result = await requestPrediction(blob, filename);
      setPrediction(result);
      setCurrentStep(4);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Prediction failed.';
      setError(message);
      setCurrentStep(0);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setIsProcessing(false);
    setCurrentStep(0);
    setVideoFile(null);
    setPrediction(null);
    setError(null);
    stopCamera();
  };

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar onReset={handleReset} />

      <main className="flex-1 overflow-auto">
        <div className="mx-auto max-w-7xl p-8">
          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="mb-2 text-3xl">Indian Sign Language Translator</h1>
                <p className="text-gray-600">Real-time ISL to English translation</p>
              </div>
              <div className="flex gap-3">
                <button className="flex items-center gap-2 rounded-lg border border-gray-300 bg-white px-4 py-2 hover:bg-gray-50">
                  <span>Language</span>
                </button>
                <button className="rounded-lg bg-black px-4 py-2 text-white hover:bg-gray-800">
                  Export Results
                </button>
              </div>
            </div>
          </div>

          {/* Video Input Section */}
          <VideoInput
            onVideoUpload={handleVideoUpload}
            videoFile={videoFile}
            isProcessing={isProcessing}
            onToggleCamera={handleToggleCamera}
            cameraStream={cameraStream}
            previewUrl={previewUrl}
          />

          {/* Processing Pipeline */}
          <ProcessingPipeline currentStep={currentStep} isProcessing={isProcessing} />

          {/* Keypoints Display */}
          <KeypointsDisplay />

          {/* Translation Output */}
          <TranslationOutput
            currentStep={currentStep}
            isProcessing={isProcessing}
            onStartTranslation={handleStartTranslation}
            onReset={handleReset}
            canStart={Boolean(videoFile || cameraStream)}
            prediction={prediction}
            error={error}
          />
        </div>
      </main>
    </div>
  );
}
