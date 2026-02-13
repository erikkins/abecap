import React, { useState } from 'react';
import { Clock, Zap, X, CreditCard } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function SubscriptionBanner() {
  const { user, hasValidSubscription, trialDaysRemaining, fetchWithAuth } = useAuth();
  const [dismissed, setDismissed] = useState(false);
  const [loading, setLoading] = useState(false);

  if (!user) return null;

  const subStatus = user.subscription?.status;
  const isActive = subStatus === 'active';
  const isPastDue = subStatus === 'past_due';
  const isTrialExpired = subStatus === 'trial' && trialDaysRemaining === 0;
  const isTrialExpiring = subStatus === 'trial' && trialDaysRemaining <= 3 && trialDaysRemaining > 0;

  // Active subscribers: show nothing (manage subscription is in the user menu)
  // Unless they have cancel_at_period_end set
  const isCanceling = isActive && user.subscription?.cancel_at_period_end;

  // Don't show banner for active (non-canceling) subscribers or dismissed
  if (dismissed || (isActive && !isCanceling)) return null;

  // Only show for: trial expiring, trial expired, past_due, or canceling
  if (!isTrialExpired && !isTrialExpiring && !isPastDue && !isCanceling) return null;

  const handleUpgrade = async (plan = null) => {
    setLoading(true);
    try {
      const selectedPlan = plan || localStorage.getItem('rigacap_selected_plan') || 'monthly';

      const response = await fetchWithAuth(`${API_URL}/api/billing/create-checkout`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ plan: selectedPlan }),
      });

      if (response.ok) {
        const data = await response.json();
        window.location.href = data.checkout_url;
      } else {
        const error = await response.json().catch(() => ({}));
        alert(error.detail || 'Failed to create checkout session');
      }
    } catch (err) {
      console.error('Checkout error:', err);
      alert('Failed to start checkout. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleManageSubscription = async () => {
    setLoading(true);
    try {
      const response = await fetchWithAuth(`${API_URL}/api/billing/portal`, {
        method: 'POST',
      });

      if (response.ok) {
        const data = await response.json();
        window.location.href = data.portal_url;
      } else {
        const error = await response.json().catch(() => ({}));
        alert(error.detail || 'Failed to open billing portal');
      }
    } catch (err) {
      console.error('Portal error:', err);
      alert('Failed to open billing portal. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const getBannerStyle = () => {
    if (isCanceling) return 'bg-orange-50 border-orange-200';
    if (isTrialExpired || isPastDue) return 'bg-red-50 border-red-200';
    if (isTrialExpiring) return 'bg-yellow-50 border-yellow-200';
    return 'bg-blue-50 border-blue-200';
  };

  const getTextStyle = () => {
    if (isCanceling) return 'text-orange-800';
    if (isTrialExpired || isPastDue) return 'text-red-800';
    if (isTrialExpiring) return 'text-yellow-800';
    return 'text-blue-800';
  };

  const getMessage = () => {
    if (isCanceling) {
      const endDate = user.subscription?.current_period_end;
      const formatted = endDate ? new Date(endDate).toLocaleDateString() : 'soon';
      return `Your subscription is set to cancel on ${formatted}. You can resubscribe anytime.`;
    }
    if (isPastDue) return 'Your payment failed. Please update your payment method to continue.';
    if (isTrialExpired) return 'Your free trial has ended. Upgrade to continue using RigaCap.';
    if (trialDaysRemaining === 1) return 'Your free trial ends tomorrow! Upgrade now to keep your access.';
    return `Your free trial ends in ${trialDaysRemaining} days. Don't miss out on premium features.`;
  };

  // Past due: show "Update Payment" button (goes to portal)
  if (isPastDue) {
    return (
      <div className={`border rounded-lg p-4 mb-6 ${getBannerStyle()}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Zap className="text-red-600 flex-shrink-0" size={24} />
            <p className={`font-medium ${getTextStyle()}`}>{getMessage()}</p>
          </div>
          <button
            onClick={handleManageSubscription}
            disabled={loading}
            className="px-4 py-2 bg-red-600 text-white font-medium rounded-lg hover:bg-red-700 transition-all disabled:opacity-50 flex items-center gap-2"
          >
            <CreditCard size={16} />
            {loading ? 'Loading...' : 'Update Payment'}
          </button>
        </div>
      </div>
    );
  }

  // Canceling: show resubscribe option
  if (isCanceling) {
    return (
      <div className={`border rounded-lg p-4 mb-6 ${getBannerStyle()}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Clock className="text-orange-600 flex-shrink-0" size={24} />
            <p className={`font-medium ${getTextStyle()}`}>{getMessage()}</p>
          </div>
          <button
            onClick={handleManageSubscription}
            disabled={loading}
            className="px-4 py-2 bg-orange-600 text-white font-medium rounded-lg hover:bg-orange-700 transition-all disabled:opacity-50"
          >
            {loading ? 'Loading...' : 'Resubscribe'}
          </button>
        </div>
      </div>
    );
  }

  // Trial expiring/expired: show upgrade buttons
  return (
    <div className={`border rounded-lg p-4 mb-6 ${getBannerStyle()}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {isTrialExpired ? (
            <Zap className="text-red-600 flex-shrink-0" size={24} />
          ) : (
            <Clock className="text-yellow-600 flex-shrink-0" size={24} />
          )}
          <div>
            <p className={`font-medium ${getTextStyle()}`}>
              {getMessage()}
            </p>
            <p className="text-sm text-gray-600 mt-1">
              $20/month or $200/year (2 months free)
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => handleUpgrade('monthly')}
            disabled={loading}
            className="px-3 py-2 border border-gray-300 text-gray-700 font-medium rounded-lg hover:bg-gray-50 transition-all disabled:opacity-50"
          >
            Monthly
          </button>
          <button
            onClick={() => handleUpgrade('annual')}
            disabled={loading}
            className="px-3 py-2 bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-medium rounded-lg hover:from-blue-700 hover:to-indigo-700 transition-all disabled:opacity-50"
          >
            {loading ? 'Loading...' : 'Annual (Save $40)'}
          </button>
          {!isTrialExpired && (
            <button
              onClick={() => setDismissed(true)}
              className="p-2 text-gray-400 hover:text-gray-600"
            >
              <X size={20} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
