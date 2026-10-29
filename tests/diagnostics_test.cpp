#include "diagnostics.h"
#include <stdexcept>

static void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    using namespace diagnostics;
    Record r{};
    r.flags=(1u<<1)|(1u<<2)|(1u<<5);
    for (unsigned i=0; i<RECENT_ACTIONS; ++i) r.recent[i]=i+3;
    const Encoding encoding(r);
    const auto full=encoding.retained();
    for (unsigned w=0; w<HDC_WORDS; ++w)
        check(full[w]==encode_word(r,w), "full encoding differs from production");
    // Independently sum retained tokens for every ablation. Removing a false bit
    // must remove its zero-state token rather than silently leave it unchanged.
    for (unsigned removed=0; removed<FEATURES; ++removed) {
        const auto actual=encoding.retained(ALL ^ (1u<<removed));
        for (unsigned w=0; w<HDC_WORDS; ++w) {
            uint32_t expected=0;
            for (unsigned bit=0; bit<32; ++bit) {
                unsigned votes=0;
                for (unsigned f=0; f<FEATURES; ++f) if (f!=removed)
                    votes+=(hyperword(feature_token(r,f),w)>>bit)&1u;
                if (2*votes>=FEATURES-1) expected|=1u<<bit;
            }
            check(actual[w]==expected, "feature omission or majority denominator is incorrect");
        }
    }
    check(encoding.retained(ALL ^ 1u)!=full, "zero-state feature was not removed");
    bool rejected=false;
    try { encoding.retained(0); } catch (const std::invalid_argument&) { rejected=true; }
    check(rejected,"empty feature mask must be rejected");

    auto metadata_changed=r;
    metadata_changed.label=1; metadata_changed.split=3; metadata_changed.prefix_len=999;
    check(Encoding(metadata_changed).retained()==full, "outcome, split or length leaked into encoding");
    Vector zero{}, ones{}; ones.fill(0xffffffffu);
    auto d=distances(zero,zero.data(),ones.data());
    check(d.safe==0 && d.unsafe==1024 && d.margin()==-1024 && !d.prediction(),"SAFE distance or margin wrong");
    d=distances(ones,zero.data(),ones.data());
    check(d.safe==1024 && d.unsafe==0 && d.margin()==1024 && d.prediction(),"UNSAFE distance or margin wrong");
    d=distances(ones,zero.data(),zero.data());
    check(d.margin()==0 && !d.prediction(),"ties must predict SAFE");
    check(std::string(length_bucket(9))=="0-9" && std::string(length_bucket(10))=="10-16" &&
          std::string(length_bucket(16))=="10-16" && std::string(length_bucket(17))=="17-32" &&
          std::string(length_bucket(32))=="17-32" && std::string(length_bucket(33))=="33-64" &&
          std::string(length_bucket(64))=="33-64" && std::string(length_bucket(65))=="65-127" &&
          std::string(length_bucket(127))=="65-127" && std::string(length_bucket(128))=="128-256" &&
          std::string(length_bucket(256))=="128-256" && std::string(length_bucket(257))=="257-512" &&
          std::string(length_bucket(512))=="257-512" && std::string(length_bucket(513))=="513+",
          "incorrect length boundary");
    for (bool reverse : {false,true}) {
        const auto reordered=reorder_recent(r,reverse);
        auto original=r;
        auto sorted=reordered;
        std::sort(original.recent,original.recent+RECENT_ACTIONS);
        std::sort(sorted.recent,sorted.recent+RECENT_ACTIONS);
        check(std::equal(original.recent,original.recent+RECENT_ACTIONS,sorted.recent),
              "order control changed the action multiset");
        check(reordered.flags==r.flags,"order control changed observed state");
        const auto repeat=reorder_recent(metadata_changed,reverse);
        check(std::equal(repeat.recent,repeat.recent+RECENT_ACTIONS,reordered.recent),
              "order control uses outcome or metadata");
    }
    check(std::string(observed_cohort(r))=="observed_sensitive_untrusted_external_unapproved",
          "observed cohort incorrect");
    check(std::string(disagreement(true,true,false))=="hdc_correct_rules_wrong" &&
          std::string(disagreement(false,true,false))=="rules_correct_hdc_wrong", "disagreement category wrong");
    Intervention a;
    a.add(true,false,true); a.add(false,false,true); a.add(true,true,true);
    check(a.flips==2 && a.fixed==1 && a.broken==1 && a.model.tp==2 && a.model.fp==1,
          "intervention change accounting incorrect");
    std::puts("PASS: diagnostic encoder, omissions, distances, ties, metadata isolation, length bins and order controls");
}
