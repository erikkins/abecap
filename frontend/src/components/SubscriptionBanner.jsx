import React, { useState } from 'react';
import { Clock, Zap, X } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function SubscriptionBanner() {
  const { user, hasValidSubscription, trialDaysRemaining, fetchWithAuth } = useAuth();
  const [dismissed, setDismissed] = useState(false);
  const [loading, setLoading] = useState(false);

  // Don't show if dismissed, no user, or subscription is active
  if (dismissed || !user || user.subscription?.status === 'active') {
    return null;
  }

  const isTrialExpired = user.subscription?.status === 'trial' && trialDaysRemaining === 0;
  const isTrialExpiring = user.subscription?.status === 'trial' && trialDaysRemaining <= 3 && trialDaysRemaining > 0;

  // Show banner for expired/expiring trials or past_due
  if (!isTrialExpired && !isTrialExpiring && user.subscription?.status !== 'past_due') {
    return null;
  }

  const handleUpgrade = async (plan = null) => {
    setLoading(true);
    try {
      // Use passed plan, stored preference, or default to monthly
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
        const error = await response.json();
        alert(error.detail || 'Failed to create checkout session');
      }
    } catch (err) {
      console.error('Checkout error:', err);
      alert('Failed to start checkout. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const getBannerStyle = () => {
    if (isTrialExpired || user.subscription?.status === 'past_due') {
      return 'bg-red-50 border-red-200';
    }
    if (isTrialExpiring) {
      return 'bg-yellow-50 border-yellow-200';
    }
    return 'bg-blue-50 border-blue-200';
  };

  const getTextStyle = () => {
    if (isTrialExpired || user.subscription?.status === 'past_due') {
      return 'text-red-800';
    }
    if (isTrialExpiring) {
      return 'text-yellow-800';
    }
    return 'text-blue-800';
  };

  const getMessage = () => {
    if (user.subscription?.status === 'past_due') {
      return 'Your payment failed. Please update your payment method to continue.';
    }
    if (isTrialExpired) {
      return 'Your free trial has ended. Upgrade to continue using RigaCap.';
    }
    if (trialDaysRemaining === 1) {
      return 'Your free trial ends tomorrow! Upgrade now to keep your access.';
    }
    return `Your free trial ends in ${trialDaysRemaining} days. Don't miss out on premium features.`;
  };

  return (
    <div className={`border rounded-lg p-4 mb-6 ${getBannerStyle()}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {isTrialExpired || user.subscription?.status === 'past_due' ? (
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
          {!isTrialExpired && user.subscription?.status !== 'past_due' && (
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
