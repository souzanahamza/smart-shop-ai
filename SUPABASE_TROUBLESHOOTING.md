# Supabase Troubleshooting Guide

## Common Issues and Solutions

### Issue: Tables are empty in Supabase Dashboard

#### 1. **Check RLS (Row Level Security) Policies**

The most common issue is that RLS policies are blocking inserts. 

**Solution A: Use Service Role Key (Recommended for testing)**
- Go to Supabase Dashboard → Settings → API
- Copy the `service_role` key (NOT the `anon` key)
- Update your Streamlit secrets with the service_role key
- **Warning**: Service role key bypasses RLS. Only use for testing or trusted environments.

**Solution B: Fix RLS Policies**
Run this SQL in Supabase SQL Editor:

```sql
-- Drop existing policies
DROP POLICY IF EXISTS "Allow all operations on interaction_logs" ON interaction_logs;
DROP POLICY IF EXISTS "Allow all operations on chat_history" ON chat_history;

-- Create new policies that allow all operations
CREATE POLICY "Allow all operations on interaction_logs" ON interaction_logs
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Allow all operations on chat_history" ON chat_history
    FOR ALL USING (true) WITH CHECK (true);
```

#### 2. **Verify Tables Exist**

Run this SQL to check if tables exist:

```sql
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN ('interaction_logs', 'chat_history');
```

#### 3. **Check Streamlit Secrets**

Make sure in Streamlit Cloud → Settings → Secrets, you have:

```toml
SUPABASE_URL = "https://savtznsvnkourzequbik.supabase.co"
SUPABASE_KEY = "your_key_here"
```

#### 4. **Test Connection**

The app now includes a debug panel. Look for "🔍 Supabase Connection Status" at the top of the app. It will show:
- ✅ Connection status
- ✅ Table accessibility
- ✅ Insert capability
- ❌ Any errors

#### 5. **Check App Logs**

In Streamlit Cloud:
- Go to your app → Settings → View logs
- Look for Supabase-related errors

### Issue: Data not saving

1. **Check the debug panel** in the app for errors
2. **Verify your Supabase key** has the right permissions
3. **Check RLS policies** (see above)
4. **Try using service_role key** for testing

### Issue: Can't see tables in Dashboard

1. Make sure you ran the SQL schema
2. Refresh the Table Editor page
3. Check if you're in the correct project

### Quick Test

Run this SQL in Supabase SQL Editor to manually insert test data:

```sql
-- Test insert into interaction_logs
INSERT INTO interaction_logs (
    original_query, 
    optimized_query, 
    llm_response_text
) VALUES (
    'test query',
    'test optimized',
    'test response'
);

-- Test insert into chat_history
INSERT INTO chat_history (
    session_id,
    message_index,
    role,
    content
) VALUES (
    'test-session-123',
    0,
    'user',
    'test message'
);

-- Check if data was inserted
SELECT * FROM interaction_logs ORDER BY created_at DESC LIMIT 5;
SELECT * FROM chat_history ORDER BY created_at DESC LIMIT 5;
```

If these inserts work, the issue is with the app connection. If they don't work, the issue is with RLS policies.

