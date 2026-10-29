//! Isolated full-history experiment. Does not alter the v0 grammar or RNG stream.
use super::Rng;
use std::collections::BTreeMap;
use std::fs::{create_dir_all, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;

const PRIVATE: u32 = 1;
const EXTERNAL: u32 = 2;
const APPROVAL: u32 = 4;
const DERIVED: u32 = 8;
const SENSITIVE: u32 = 16;
const LOCAL: u32 = 32;
const UNTRUSTED: u32 = 64;
const FILLER: u32 = 128;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Event {
    action: u32, object: u32, source: u32, flags: u32, step: u32,
    dependency_begin: u32, dependency_count: u32,
}
impl Event {
    fn words(&self) -> [u32; 8] {
        [self.action, self.object, self.source, self.flags, self.step,
         self.dependency_begin, self.dependency_count, 0]
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct History { events: Vec<Event>, edges: Vec<[u32; 3]>, label: u32 }
#[derive(Default, Clone, Copy)]
struct Object { sensitive: bool, approved: bool, writer: u32, approval: Option<u32> }

fn history(length: usize, seed: u64, unsafe_order: bool) -> History {
    let mut rng = Rng(seed);
    let mut events = Vec::with_capacity(length);
    // Filler objects occupy a separate namespace and never modify payload objects 1/2.
    for step in 0..length {
        let action = [6, 7, 8, 15, 17][rng.range(5)];
        let flags = FILLER | match action {
            8 => PRIVATE | SENSITIVE,
            7 => EXTERNAL,
            15 => LOCAL,
            _ => 0,
        };
        events.push(Event { action, object: 1000 + rng.range(32) as u32, source: 0,
            flags, step: step as u32, dependency_begin: 0, dependency_count: 0 });
    }
    if length >= 5 {
        let mut place = |step: usize, action, object, source, flags| {
            events[step] = Event { action, object, source, flags, step: step as u32,
                dependency_begin: 0, dependency_count: 0 };
        };
        place(0, 1, 1, 0, PRIVATE | SENSITIVE);
        place(1, 4, 2, 1, DERIVED | SENSITIVE);
        let (early, late) = (2 + (length-5) / 3, 3 + 2 * (length-5) / 3);
        if unsafe_order {
            place(early, 11, 2, 0, APPROVAL);
            place(late, 4, 2, 1, DERIVED | SENSITIVE);
        } else {
            place(early, 4, 2, 1, DERIVED | SENSITIVE);
            place(late, 11, 2, 0, APPROVAL);
        }
        place(length - 1, 13, 2, 0, EXTERNAL | UNTRUSTED | SENSITIVE);
    }
    let mut edges = Vec::new();
    let mut objects: BTreeMap<u32, Object> = BTreeMap::new();
    for e in &mut events {
        e.dependency_begin = edges.len() as u32;
        match e.action {
            1 | 8 => { objects.insert(e.object, Object { sensitive: true,
                writer: e.step, ..Object::default() }); }
            4 => {
                let source = objects[&e.source];
                edges.push([e.step, source.writer, e.source]);
                // A rewrite creates a new object version and invalidates old approval.
                objects.insert(e.object, Object { sensitive: source.sensitive,
                    writer: e.step, ..Object::default() });
            }
            11 => {
                let object = objects.get_mut(&e.object).unwrap();
                edges.push([e.step, object.writer, e.object]);
                object.approved = true;
                object.approval = Some(e.step);
            }
            13 => {
                let object = objects[&e.object];
                edges.push([e.step, object.writer, e.object]);
                if let Some(approval) = object.approval { edges.push([e.step, approval, e.object]); }
            }
            15 => { objects.insert(e.object, Object { writer: e.step, ..Object::default() }); }
            _ => {}
        }
        e.dependency_count = edges.len() as u32 - e.dependency_begin;
    }
    // Hypothetical final external transfer is AFTER the observed cutoff. It is
    // never appended to events. Outcome is stored in a physically separate table.
    let payload = objects.get(&2).copied().unwrap_or_default();
    let label = u32::from(length >= 5 && payload.sensitive && !payload.approved);
    History { events, edges, label }
}
fn words(writer: &mut impl Write, values: &[u32]) -> io::Result<()> {
    for value in values { writer.write_all(&value.to_le_bytes())?; }
    Ok(())
}
fn option<'a>(args: &'a [String], name: &str, default: &'a str) -> io::Result<&'a str> {
    match args.iter().position(|arg| arg == name) {
        None => Ok(default),
        Some(i) => args.get(i+1).filter(|s| !s.starts_with("--")).map(String::as_str)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, format!("missing {name} value"))),
    }
}
pub fn run(args: &[String]) -> io::Result<()> {
    let invalid = |message| io::Error::new(io::ErrorKind::InvalidInput, message);
    let lengths: Vec<usize> = option(args, "--temporal-lengths", "32,1024,4096,32768")?
        .split(',').map(|s| s.parse::<usize>().map_err(|_| invalid("invalid temporal length")))
        .collect::<io::Result<_>>()?;
    let seed = option(args, "--seed", "42")?.parse::<u64>().map_err(|_| invalid("invalid seed"))?;
    let pairs = option(args, "--temporal-pairs", "1")?.parse::<usize>().map_err(|_| invalid("invalid pair count"))?;
    if pairs == 0 || lengths.iter().any(|n| *n > 1_000_000) {
        return Err(invalid("pairs must be positive; each history is limited to 1000000 events"));
    }
    let count = lengths.len().checked_mul(pairs).and_then(|n| n.checked_mul(2))
        .filter(|n| *n <= 10000).ok_or_else(|| invalid("too many histories"))?;
    let total = lengths.iter().try_fold(0usize, |a,n| a.checked_add(*n))
        .and_then(|n| n.checked_mul(pairs)).and_then(|n| n.checked_mul(2))
        .filter(|n| *n <= 2_000_000).ok_or_else(|| invalid("too many events (limit 2000000)"))?;
    let mut histories = Vec::with_capacity(count);
    for (li, length) in lengths.iter().enumerate() {
        for pair in 0..pairs {
            let pair_seed = seed.wrapping_add((li as u64).wrapping_mul(0x9e3779b97f4a7c15)).wrapping_add(pair as u64);
            histories.push(history(*length, pair_seed, false));
            histories.push(history(*length, pair_seed, true));
        }
    }
    let edge_count: usize = histories.iter().map(|h| h.edges.len()).sum();
    let descriptor_offset = 64u32;
    let event_offset = descriptor_offset + count as u32 * 32;
    let edge_offset = event_offset + total as u32 * 32;
    let label_offset = edge_offset + edge_count as u32 * 12;
    let path = Path::new(option(args, "--out", "data/temporal_v1.bin")?);
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) { create_dir_all(parent)?; }
    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(b"HDBTMP01")?;
    words(&mut writer, &[1, 64, count as u32, total as u32, edge_count as u32,
        32, 12, 32, descriptor_offset, event_offset, edge_offset, label_offset, 16, 0])?;
    let (mut event_begin, mut edge_begin) = (0, 0);
    for (id, h) in histories.iter().enumerate() {
        words(&mut writer, &[id as u32, id as u32 / 2, event_begin, h.events.len() as u32,
            edge_begin, h.edges.len() as u32, h.events.len() as u32, 0])?;
        event_begin += h.events.len() as u32;
        edge_begin += h.edges.len() as u32;
    }
    for h in &histories { for e in &h.events { words(&mut writer, &e.words())?; } }
    for h in &histories { for edge in &h.edges { words(&mut writer, edge)?; } }
    for (id, h) in histories.iter().enumerate() {
        words(&mut writer, &[id as u32, h.label, u32::from(h.events.len() >= 5), h.label])?;
    }
    writer.flush()?;
    println!("Temporal v1: {} histories, {} observed events, {} exact dependency edges -> {}",
        count, total, edge_count, path.display());
    println!("Labels are a separate outcome table; no final transfer is exported.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn order_changes_outcome_with_same_event_multiset() {
        for n in [5, 32, 1024, 4096, 32768] {
            let safe = history(n, 42, false);
            let unsafe_h = history(n, 42, true);
            assert_eq!((safe.label, unsafe_h.label), (0, 1));
            let bag = |h: &History| { let mut events: Vec<_> = h.events.iter()
                .map(|e| (e.action,e.object,e.source,e.flags)).collect(); events.sort(); events };
            assert_eq!(bag(&safe), bag(&unsafe_h));
            assert_eq!(safe, history(n, 42, false));
            assert_eq!(safe.events.len(), n);
            for e in &safe.events {
                assert!((e.step as usize) < n);
                for edge in &safe.edges[e.dependency_begin as usize..(e.dependency_begin+e.dependency_count) as usize] {
                    assert_eq!(edge[0],e.step); assert!(edge[1]<edge[0]);
                }
            }
        }
    }
    #[test]
    fn empty_and_partial_histories_have_no_future_transfer() {
        for n in [0, 1, 2, 3, 4, 31, 33, 1025] {
            let h=history(n, 123, false);
            assert_eq!(h.events.len(),n);
            assert!(!h.events.iter().any(|e| e.action==18));
            if n<5 { assert_eq!(h.label,0); }
        }
    }
}
