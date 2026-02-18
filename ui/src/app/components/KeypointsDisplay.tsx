import { Hand, MoreHorizontal, Smile, User } from "lucide-react";

const keypointTypes = [
  { title: "Hand Landmarks", count: 21, icon: Hand },
  { title: "Pose Landmarks", count: 33, icon: User },
  { title: "Face Landmarks", count: 468, icon: Smile },
];

export function KeypointsDisplay() {
  return (
    <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4 sm:p-5">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h2 className="mb-1 text-[18px]">Extracted Keypoints</h2>
          <p className="text-xs text-gray-600">Visual representation of detected landmarks</p>
        </div>
        <button className="flex items-center gap-1 rounded-md border border-gray-200 px-2.5 py-1.5 text-xs text-gray-600 hover:bg-gray-50">
          <MoreHorizontal className="h-3.5 w-3.5" />
          <span>View Details</span>
        </button>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        {keypointTypes.map((type) => {
          const Icon = type.icon;

          return (
            <div key={type.title} className="overflow-hidden rounded-md border border-gray-200 bg-white">
              <div className="flex h-48 items-center justify-center bg-gray-100">
                <Icon className="h-8 w-8 text-gray-400" strokeWidth={1.5} />
              </div>
              <div className="border-t border-gray-200 px-3 py-2 text-center">
                <p className="text-xs text-gray-700">{type.title}</p>
                <p className="text-xs text-orange-700">{type.count} keypoints detected</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
