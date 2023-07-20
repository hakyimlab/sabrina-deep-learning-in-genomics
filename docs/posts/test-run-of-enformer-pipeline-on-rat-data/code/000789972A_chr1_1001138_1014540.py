import EnformerVCF
import kipoiseq
import numpy as np

target_interval = kipoiseq.Interval("chr1", 1001138, 1014540)
chr1_vcf = EnformerVCF.read_vcf('/home/s1mi/enformer_rat_data/BrainVCFs/chr1.vcf.gz')
haplo_1, haplo_2 = EnformerVCF.vcf_to_seq(target_interval, '000789972A', chr1_vcf)


haplo_1_enc = EnformerVCF.one_hot_encode("".join(haplo_1))[np.newaxis]
haplo_2_enc = EnformerVCF.one_hot_encode("".join(haplo_2))[np.newaxis]

prediction_1 = EnformerVCF.model.predict_on_batch(haplo_1_enc)['human'][0]
prediction_2 = EnformerVCF.model.predict_on_batch(haplo_2_enc)['human'][0]

predictions = (prediction_1 + prediction_2) / 2
print(predictions)