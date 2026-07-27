import { Navigate, Route, Routes, useLocation } from 'react-router-dom';
import { useAuthStore } from '@/store/useAuthStore';
import AuthLayout from '@/layouts/AuthLayout';
import DashboardLayout from '@/layouts/DashboardLayout';
import Login from '@/pages/Login';
import KBHub from '@/pages/KBHub';
import DocumentManager from '@/pages/DocumentManager';
import KnowledgeQA from '@/pages/KnowledgeQA';
import KnowledgeGraph from '@/pages/KnowledgeGraph';
import RetrievalEval from '@/pages/RetrievalEval';
import Admin from '@/pages/Admin';

/** Guards routes that require an authenticated session. */
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const location = useLocation();
  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }
  return <>{children}</>;
}

/**
 * Root application router.
 * - `/login` uses the split AuthLayout (brand + form).
 * - All other routes are protected and share the DashboardLayout shell.
 */
export default function App() {
  return (
    <Routes>
      <Route
        path="/login"
        element={
          <AuthLayout>
            <Login />
          </AuthLayout>
        }
      />
      <Route
        path="/"
        element={
          <ProtectedRoute>
            <DashboardLayout />
          </ProtectedRoute>
        }
      >
        <Route index element={<Navigate to="/qa" replace />} />
        <Route path="hub" element={<KBHub />} />
        <Route path="documents" element={<DocumentManager />} />
        <Route path="qa" element={<KnowledgeQA />} />
        <Route path="graph" element={<KnowledgeGraph />} />
        <Route path="evaluation" element={<RetrievalEval />} />
        <Route path="admin" element={<Admin />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
