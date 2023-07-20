bcftools index -t /home/s1mi/enformer_rat_data/Brain.rn7.vcf.gz

# Get list of samples
bcftools query -l /home/s1mi/enformer_rat_data/Brain.rn7.vcf.gz > /home/s1mi/enformer_rat_pipeline/metadata/individuals.txt


# Split VCF by chromosome
vcf_in=/home/s1mi/enformer_rat_data/Brain.rn7.vcf.gz

vcf_out_prefix=/home/s1mi/enformer_rat_data/BrainVCFs/chr

for i in {1..20}
do
    echo "Working on chromosome ${i}..."
    bcftools view ${vcf_in} --regions ${i} -o ${vcf_out_prefix}${i}.vcf.gz -Oz
done


# Index VCFs
for i in {1..20}
do
    echo "Indexing chromosome ${i}..."
    bcftools index -t /home/s1mi/enformer_rat_data/BrainVCFs/chr${i}.vcf.gz
done
