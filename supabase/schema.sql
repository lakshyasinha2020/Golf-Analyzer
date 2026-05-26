-- SwingGrade Supabase setup
-- Run this in the Supabase SQL editor.

create extension if not exists "pgcrypto";

create table if not exists public.swing_videos (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  title text not null,
  source_name text not null,
  storage_path text not null unique,
  handedness text not null check (handedness in ('right', 'left')),
  swing_angle text not null check (swing_angle in ('face-on', 'down-the-line')),
  overall_score integer not null check (overall_score >= 0 and overall_score <= 100),
  analysis jsonb not null,
  created_at timestamptz not null default now()
);

create index if not exists swing_videos_user_created_idx
  on public.swing_videos (user_id, created_at desc);

create index if not exists swing_videos_overall_idx
  on public.swing_videos (overall_score desc);

alter table public.swing_videos enable row level security;

drop policy if exists "Users can insert their swing videos" on public.swing_videos;
create policy "Users can insert their swing videos"
  on public.swing_videos
  for insert
  to authenticated
  with check ((select auth.uid()) = user_id);

drop policy if exists "Users can read their swing videos" on public.swing_videos;
create policy "Users can read their swing videos"
  on public.swing_videos
  for select
  to authenticated
  using ((select auth.uid()) = user_id);

drop policy if exists "Users can update their swing videos" on public.swing_videos;
create policy "Users can update their swing videos"
  on public.swing_videos
  for update
  to authenticated
  using ((select auth.uid()) = user_id)
  with check ((select auth.uid()) = user_id);

drop policy if exists "Users can delete their swing videos" on public.swing_videos;
create policy "Users can delete their swing videos"
  on public.swing_videos
  for delete
  to authenticated
  using ((select auth.uid()) = user_id);

grant select, insert, update, delete on public.swing_videos to authenticated;

-- Limited leaderboard view for client-side rankings.
-- It exposes scores but not storage paths, source file names, or full analysis JSON.
drop view if exists public.swing_leaderboard_entries;
create view public.swing_leaderboard_entries
with (security_invoker = true) as
select
  id,
  md5(user_id::text) as player_key,
  title,
  overall_score,
  created_at,
  coalesce(
    (
      select (score_item->>'score')::integer
      from jsonb_array_elements(analysis->'categoryScores') as score_item
      where score_item->>'id' = 'tempo'
      limit 1
    ),
    0
  ) as tempo_score,
  coalesce(
    (
      select (score_item->>'score')::integer
      from jsonb_array_elements(analysis->'categoryScores') as score_item
      where score_item->>'id' = 'impact'
      limit 1
    ),
    0
  ) as impact_score
from public.swing_videos;

grant select on public.swing_leaderboard_entries to authenticated;

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('swing-videos', 'swing-videos', false, 52428800, array['video/*'])
on conflict (id) do update
set
  public = false,
  file_size_limit = 52428800,
  allowed_mime_types = array['video/*'];

drop policy if exists "Users can upload their swing videos" on storage.objects;
create policy "Users can upload their swing videos"
  on storage.objects
  for insert
  to authenticated
  with check (
    bucket_id = 'swing-videos'
    and (select auth.uid())::text = (storage.foldername(name))[1]
  );

drop policy if exists "Users can read their swing videos" on storage.objects;
create policy "Users can read their swing videos"
  on storage.objects
  for select
  to authenticated
  using (
    bucket_id = 'swing-videos'
    and (select auth.uid())::text = (storage.foldername(name))[1]
  );

drop policy if exists "Users can update their swing videos" on storage.objects;
create policy "Users can update their swing videos"
  on storage.objects
  for update
  to authenticated
  using (
    bucket_id = 'swing-videos'
    and (select auth.uid())::text = (storage.foldername(name))[1]
  )
  with check (
    bucket_id = 'swing-videos'
    and (select auth.uid())::text = (storage.foldername(name))[1]
  );

drop policy if exists "Users can delete their swing videos" on storage.objects;
create policy "Users can delete their swing videos"
  on storage.objects
  for delete
  to authenticated
  using (
    bucket_id = 'swing-videos'
    and (select auth.uid())::text = (storage.foldername(name))[1]
  );
