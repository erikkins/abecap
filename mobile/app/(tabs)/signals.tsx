/**
 * Signals list — all buy signals with search/filter.
 */

import React, { useMemo, useState } from 'react';
import {
  FlatList,
  RefreshControl,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { useRouter } from 'expo-router';
import { useDashboard } from '@/hooks/useSignals';
import SignalCard from '@/components/SignalCard';
import { Colors, FontSize, Spacing } from '@/constants/theme';

export default function SignalsScreen() {
  const { data, isLoading, refresh } = useDashboard();
  const router = useRouter();
  const [search, setSearch] = useState('');

  const signals = useMemo(() => {
    const all = data?.buy_signals || [];
    if (!search) return all;
    const q = search.toUpperCase();
    return all.filter((s) => s.symbol.includes(q));
  }, [data?.buy_signals, search]);

  return (
    <View style={styles.container}>
      {/* Search */}
      <View style={styles.searchBar}>
        <TextInput
          style={styles.searchInput}
          placeholder="Search by symbol..."
          placeholderTextColor={Colors.textMuted}
          value={search}
          onChangeText={setSearch}
          autoCapitalize="characters"
        />
      </View>

      {/* Signal List */}
      <FlatList
        data={signals}
        keyExtractor={(item) => item.symbol}
        renderItem={({ item }) => (
          <SignalCard
            signal={item}
            onPress={() => router.push(`/signal/${item.symbol}`)}
          />
        )}
        contentContainerStyle={styles.list}
        refreshControl={
          <RefreshControl
            refreshing={isLoading}
            onRefresh={refresh}
            tintColor={Colors.gold}
          />
        }
        ListEmptyComponent={
          <View style={styles.empty}>
            <Text style={styles.emptyText}>
              {search ? 'No signals match your search' : 'No signals available'}
            </Text>
          </View>
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  searchBar: {
    padding: Spacing.md,
    paddingBottom: 0,
  },
  searchInput: {
    backgroundColor: Colors.card,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: Colors.cardBorder,
    padding: Spacing.md,
    color: Colors.textPrimary,
    fontSize: FontSize.md,
  },
  list: {
    padding: Spacing.md,
    paddingBottom: Spacing.xl,
  },
  empty: {
    alignItems: 'center',
    padding: Spacing.xl,
  },
  emptyText: {
    color: Colors.textMuted,
    fontSize: FontSize.md,
  },
});
