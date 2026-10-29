"""Self-contained Python fixture for validating binary format and CPU/CUDA code.

Not the canonical generator. Use `cargo run` for actual experiments.
"""
import argparse
import random
import struct
from pathlib import Path

MAGIC = b'HDBSAFE1'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='data/fixture.bin')
    p.add_argument('--records',type=int,default=800)
    args=p.parse_args()
    rng=random.Random(42)
    rows=[]
    for i in range(args.records):
        split=i//max(1,(args.records//4))
        split=min(split,3)
        label=(i%2)==0
        neg=rng.randrange(4)
        # All records have sensible input bit states and known ground truth;
        # purposely include uncertain negative controls.
        sensitive=bool(label or neg != 0)
        untrusted=bool(label or neg != 1)
        external=bool(label or neg != 3)
        approved=bool(not label and neg == 2)
        flags=(int(sensitive)<<1)|(int(untrusted)<<2)|(int(approved)<<4)|(
            int(external)<<5)|(1<<6)|(int(sensitive)<<0)
        if rng.random()<0.33: flags|=1<<7  # unrelated private read
        if rng.random()<0.30: flags|=1<<8  # web search
        if rng.random()>.9:
            flags^=1<<2  # noisy destination observation, not oracle state
        steps=(150+rng.randrange(300)) if split==3 else (10+rng.randrange(50))
        recent=[rng.choice([3,4,5,6,7,8,15,16,17]) for _ in range(8)]
        rows.append((split,label,flags,steps,*recent))
    output=Path(args.out)
    output.parent.mkdir(exist_ok=True,parents=True)
    with output.open('wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('<IIII',1,len(rows),1024,12))
        for row in rows:
            f.write(struct.pack('<12I',*row))
    assert output.stat().st_size == 24+48*len(rows)
    print(f'Wrote independent binary format fixture {output} ({len(rows)} records)')

if __name__=='__main__': main()
