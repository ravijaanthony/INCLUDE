interface SidebarProps {
  onReset: () => void;
}

export function Sidebar({ onReset }: SidebarProps) {
  return (
    <aside className="w-64 bg-white border-r border-gray-200 p-6 flex flex-col">
      {/* Logo */}
      <div className="flex items-center gap-3 mb-8">
        <div className="w-10 h-10 bg-black rounded-lg flex items-center justify-center">
          <span className="text-white text-xl">🤟</span>
        </div>
        <span className="font-semibold">LumiSign</span>
      </div>

      {/* Navigation */}
      <nav className="flex-1">
        <div className="space-y-2">
          <button 
            onClick={onReset}
            className="w-full flex items-center gap-3 px-4 py-3 rounded-lg bg-gray-100 hover:bg-gray-200 text-left"
          >
            <span>📹</span>
            <span>New Translation</span>
          </button>
          
          <button className="w-full flex items-center gap-3 px-4 py-3 rounded-lg hover:bg-gray-100 text-left text-gray-600">
            <span>🕐</span>
            <span>History</span>
          </button>
          
          <button className="w-full flex items-center gap-3 px-4 py-3 rounded-lg hover:bg-gray-100 text-left text-gray-600">
            <span>🔖</span>
            <span>Saved</span>
          </button>
          
          <button className="w-full flex items-center gap-3 px-4 py-3 rounded-lg hover:bg-gray-100 text-left text-gray-600">
            <span>📊</span>
            <span>Analytics</span>
          </button>
        </div>

        {/* Resources Section */}
        <div className="mt-8">
          <div className="text-xs uppercase text-gray-400 mb-3 px-4">Resources</div>
          <div className="space-y-2">
            <button className="w-full flex items-center gap-3 px-4 py-3 rounded-lg hover:bg-gray-100 text-left text-gray-600">
              <span>📚</span>
              <span>Documentation</span>
            </button>
            
            <button className="w-full flex items-center gap-3 px-4 py-3 rounded-lg hover:bg-gray-100 text-left text-gray-600">
              <span>❓</span>
              <span>Help Center</span>
            </button>
            
            <button className="w-full flex items-center gap-3 px-4 py-3 rounded-lg hover:bg-gray-100 text-left text-gray-600">
              <span>⚙️</span>
              <span>Settings</span>
            </button>
          </div>
        </div>
      </nav>

      {/* User Profile */}
      <div className="pt-6 border-t border-gray-200">
        <div className="flex items-center gap-3 px-4 py-3">
          <div className="w-10 h-10 bg-gray-300 rounded-full flex items-center justify-center">
            <span>👤</span>
          </div>
          <div className="flex-1">
            <div className="font-medium">Priya Sharma</div>
            <div className="text-sm text-gray-500">priya@example.com</div>
          </div>
          <button className="text-gray-400 hover:text-gray-600">⋮</button>
        </div>
      </div>
    </aside>
  );
}
