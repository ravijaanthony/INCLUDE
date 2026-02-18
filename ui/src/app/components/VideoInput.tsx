import { useEffect, useRef } from 'react';

interface VideoInputProps {
  onVideoUpload: (file: File) => void;
  videoFile: File | null;
  isProcessing: boolean;
  onToggleCamera: () => void;
  cameraStream: MediaStream | null;
  previewUrl: string | null;
}

export function VideoInput({
  onVideoUpload,
  videoFile,
  isProcessing,
  onToggleCamera,
  cameraStream,
  previewUrl,
}: VideoInputProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const previewRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const video = previewRef.current;
    if (!video) return;

    if (cameraStream) {
      video.srcObject = cameraStream;
      video.muted = true;
      void video.play();
      return;
    }

    if (previewUrl) {
      video.srcObject = null;
      video.src = previewUrl;
      void video.play();
      return;
    }

    video.srcObject = null;
    video.removeAttribute('src');
    video.load();
  }, [cameraStream, previewUrl]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onVideoUpload(file);
    }
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6 mb-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-xl mb-1">Video Input</h2>
          <p className="text-sm text-gray-600">Upload a video or use your camera to capture sign language</p>
        </div>
        <div className="flex gap-2">
          <button 
            onClick={onToggleCamera}
            disabled={isProcessing}
            className="px-4 py-2 border border-gray-300 rounded-lg bg-white hover:bg-gray-50 flex items-center gap-2 disabled:cursor-not-allowed disabled:opacity-60"
          >
            <span>📹</span>
            Use Camera
          </button>
          <button 
            onClick={() => fileInputRef.current?.click()}
            disabled={isProcessing}
            className="px-4 py-2 bg-black text-white rounded-lg hover:bg-gray-800 flex items-center gap-2 disabled:cursor-not-allowed disabled:opacity-60"
          >
            <span>⬆</span>
            Upload Video
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            className="hidden"
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Original Video Feed */}
        <div className="bg-gray-900 rounded-lg p-8 aspect-video flex flex-col items-center justify-center">
          {cameraStream || previewUrl ? (
            <video
              ref={previewRef}
              className="h-full w-full rounded-lg object-cover"
              playsInline
            />
          ) : (
            <>
              <div className="text-gray-600 mb-3">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <rect x="2" y="7" width="20" height="14" rx="2" ry="2"></rect>
                  <polyline points="23 7 16 12 23 17"></polyline>
                </svg>
              </div>
              <div className="text-gray-500 text-center">
                <div className="font-medium mb-1">Original Video Feed</div>
                <div className="text-sm">Camera inactive</div>
              </div>
            </>
          )}
          {videoFile && (
            <div className="mt-4 text-white text-sm">
              ✓ {videoFile.name}
            </div>
          )}
          <div className="mt-6 text-gray-600 text-sm">
            Duration: 00:00
          </div>
        </div>

        {/* Enhanced Video */}
        <div className="bg-gray-900 rounded-lg p-8 aspect-video flex flex-col items-center justify-center">
          <div className="text-gray-600 mb-3">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <circle cx="12" cy="12" r="3"></circle>
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="12" y1="2" x2="12" y2="4"></line>
              <line x1="12" y1="20" x2="12" y2="22"></line>
              <line x1="4.93" y1="4.93" x2="6.34" y2="6.34"></line>
              <line x1="17.66" y1="17.66" x2="19.07" y2="19.07"></line>
              <line x1="2" y1="12" x2="4" y2="12"></line>
              <line x1="20" y1="12" x2="22" y2="12"></line>
              <line x1="6.34" y1="17.66" x2="4.93" y2="19.07"></line>
              <line x1="19.07" y1="4.93" x2="17.66" y2="6.34"></line>
            </svg>
          </div>
          <div className="text-gray-500 text-center">
            <div className="font-medium mb-1">Enhanced Video</div>
            <div className="text-sm">Brightness correction applied</div>
          </div>
          <div className="mt-6 flex gap-8 text-sm">
            <div className="text-gray-600">
              Resolution: --
            </div>
            <div className="text-gray-600">
              Enhancement: Pending
            </div>
            <div className="text-gray-600">
              Quality: --
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
