#include <unordered_map>
#include <cstdio>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <deque>
#include <algorithm>
#include <ctime>
#include <cassert>
#include <cmath>
#include <limits.h>
#include "utilities.h"
using namespace std;

enum Stop {STOP, CONTINUE};
enum Kids {NO_KIDS, ONE_KID, TWO_OR_MORE};
enum Seal {R, L, F};
enum Dir  {LEFT, RIGHT};
enum constr_type {RULE,CHUNK};
int  constr_count = 1;
struct token
{
	int word;
	int tag;	//gold tag
	int ctag;   //coarse tag
	int subtag; //refined tag
};

struct dep
{
	int head_idx;
	int arg_idx;
};

bool operator==(dep &dp1, dep &dp2)
{
	return (dp1.head_idx == dp2.head_idx && dp1.arg_idx == dp2.arg_idx);
}
bool equals(dep &dp1, dep &dp2)
{
	return (dp1.head_idx == dp2.head_idx && dp1.arg_idx == dp2.arg_idx) || (dp1.arg_idx == dp2.head_idx && dp1.head_idx == dp2.arg_idx);
}
struct sentence
{
	vector<token> words;
	vector<dep>   pdeps;	//predicted dependencies
	vector<dep>   gdeps;	//gold dependencies
	vector<dep>   rdeps;	//rule based dependecies
};
class data
{
public:
	unordered_map<string,int>	  word_map;
	vector<string>			 words;
	
	//gold tagset
	unordered_map<string,int>  gtag_map;
	vector<string>			 gtags;		
	//coarse tagset
	unordered_map<string,int>  ctag_map;
	vector<string>         ctags;		
	int **rules;			//matrix of binary values encoding dependency rules
	int **neg_rules;			//matrix of binary values encoding allowable dependecies
	int ***dir_rules;

	unordered_map<string, string> config;

	vector<sentence> sents;

	int tsize;
	int wsize;
	int senlen;
	int Root;

	void load_config();
	void populate_rule_deps();
	void load_data();
	void load_tagset_map();
	void load_rules();
	void load_neg_rules();
	void load_dir_rules();
	void print_results(bool);
	void print_rule_results(bool);
	void print_subtags();
	void write_output();
	void write_stats(bool);
};

void data::load_tagset_map()
{
	vector<int> tag_map(gtags.size());
	ifstream ftmap(config["tag_set_map"].c_str());
	char line[500];
	while( !ftmap.eof() )
	{
		ftmap.getline(line,500);

		char * tok = strtok(line, ":");
		string ctag(tok); trim(ctag);
		int ctag_num;
		if(ctag_map.find(ctag) == ctag_map.end())
		{
			ctag_num = ctags.size();
			ctag_map[ctag] = ctag_num;
			ctags.push_back(ctag);
		}
		else
			ctag_num = ctag_map[ctag];
		
		tok = strtok(NULL, " ");
		
		while(tok)
		{
			string rtag(tok); trim(rtag);
			if(gtag_map.find(rtag)==gtag_map.end())
			{
				gtag_map[rtag] = gtags.size();
				gtags.push_back(rtag);
				int rtag_num = gtag_map[rtag];
				tag_map.push_back(ctag_num);
			}
			else
			{
				int rtag_num = gtag_map[rtag];
				tag_map[rtag_num]=ctag_num;
			}
			tok = strtok(NULL, " ");
		}
	}

	for(int s = 0; s < sents.size(); s++)
	{
		for(int w = 0; w < sents[s].words.size(); w++)
		{
			sents[s].words[w].ctag = tag_map[sents[s].words[w].tag];
		}
	}

}

void data::load_rules()
{
	//the rule format:
	//X -> Y    means that X generates Y
	
	alloc_arr_2D(ctags.size(),ctags.size(),rules);
	for(int i = 0; i < ctags.size(); i++)
		for(int j = 0; j < ctags.size(); j++)
			rules[i][j] = 0;
	ifstream frule(config["rules"].c_str());
	char rline[1000];
	while( !frule.eof() )
	{
		frule.getline(rline,1000);
		char *tok = strtok(rline, " ");
		string tag1(tok); trim(tag1);
		tok = strtok(NULL, " ");
		string arrow(tok); trim(arrow);
		tok = strtok(NULL, " ");
		string tag2(tok); trim(tag2);
		rules[ctag_map[tag1]][ctag_map[tag2]] = 1;
	}
}

void data::load_dir_rules()
{
	//the rule format:
	//X -> Y    means that X generates Y
	
	alloc_arr_3D(2,gtags.size(),gtags.size(),dir_rules);
	for(int i = 0; i < gtags.size(); i++)
		for(int j = 0; j < gtags.size(); j++)
		{
			dir_rules[0][i][j] = 0;
			dir_rules[1][i][j] = 0;
		}
	ifstream frule(config["dir_rules"].c_str());
	char rline[1000];
	while( !frule.eof() )
	{
		frule.getline(rline,1000);
		char *tok = strtok(rline, " ");
		string tag1(tok); trim(tag1);
		tok = strtok(NULL, " ");
		string arrow(tok); trim(arrow);
		Dir dir = arrow[0] == '<' ? LEFT : RIGHT ;
		tok = strtok(NULL, " ");
		string tag2(tok); trim(tag2);
		if (dir == RIGHT)
			dir_rules[dir][gtag_map[tag1]][gtag_map[tag2]] = 1;
		else
			dir_rules[dir][gtag_map[tag2]][gtag_map[tag1]] = 1;
	}
}

void data::load_neg_rules()
{
	alloc_arr_2D(ctags.size(),ctags.size(),neg_rules);
	for(int i = 0; i < ctags.size(); i++)
		for(int j = 0; j < ctags.size(); j++)
			neg_rules[i][j] = 0;
	ifstream frule(config["neg_rules"].c_str());
	char rline[2000];
	vector<string> tags;
	frule.getline(rline,2000);
	char *tok = strtok(rline,"\t");
	while(tok)
	{
		string tag(tok); trim(tag);
		if(tag.size()>0)
			tags.push_back(tag);
		tok = strtok(NULL,"\t");
	}
	int t = 0;
	while( !frule.eof() && t<tags.size() )
	{
		frule>>rline;
		for(int tt = 0; tt < tags.size(); tt++)
		{
			int x;
			frule>>x;
			neg_rules[ctag_map[tags[t]]][ctag_map[tags[tt]]] = x;
		}
		t++;
	}
}
struct span{int start; int end;};
void data::populate_rule_deps()
{
	for(int s = 0; s < sents.size(); s++)
	{
		sents[s].rdeps.resize(sents[s].words.size()-1);

		//find root of the sentence
		int root_idx; int j;
		for(j = 0; j < sents[s].words.size() - 1; j++)
		{
			if( gtags[sents[s].words[j].tag][0] == 'M' || gtags[sents[s].words[j].tag][0] == 'V')
			{
				root_idx = j;
				break;
			}
		}
		if( j == sents[s].words.size() - 1)
		{
			if(sents[s].words.size()>2)
				root_idx = 1;
			else
				root_idx = 0;
		}

		//chunk the sentence
		vector<span> chunks;
		for(int j = sents[s].words.size() - 2; j >= 0; j--)
		{
			if( gtags[sents[s].words[j].tag][0] == 'N' || gtags[sents[s].words[j].tag][0] == 'P')
			{
				int end = j;
				//find the whole chunk J C D
				while( j >= 0 &&
				      (gtags[sents[s].words[j].tag][0] == 'N' || 
				       gtags[sents[s].words[j].tag][0] == 'P' ||
				       gtags[sents[s].words[j].tag][0] == 'J' || 
				       gtags[sents[s].words[j].tag][0] == 'C' || 
				       gtags[sents[s].words[j].tag][0] == 'D' ) )j--;
				
				int start = j + 1;
				
				//put it in the ch_sen 
				span s; s.start = start; s.end = end;
				chunks.push_back(s);
				
				j++;
			}
			else
			{
				//just the current tag is the chunk
				span s; s.start = j; s.end = j;
				chunks.push_back(s);				
			}
		}

		//deal with each chunk
		for(int i = 0; i < chunks.size(); i++)
		{
			if(chunks[i].start < chunks[i].end)
			{
				for(int a = chunks[i].start; a < chunks[i].end; a++)
				{
					dep d; d.arg_idx = a; d.head_idx = chunks[i].end;
					sents[s].rdeps[a] = d;
					if(a == root_idx)
						root_idx = chunks[i].end;
				}

				if( chunks[i].start > 0 &&
					(strcmp( gtags[sents[s].words[chunks[i].start - 1].tag].c_str(), "IN") == 0 ||
					strcmp( gtags[sents[s].words[chunks[i].start - 1].tag].c_str(), "TO") == 0 ))
				{
					dep d; d.head_idx = chunks[i].start - 1; d.arg_idx = chunks[i].end;
					sents[s].rdeps[d.arg_idx] = d;
				}
				else if( chunks[i].end < root_idx )
				{
					dep d; d.head_idx = root_idx; d.arg_idx = chunks[i].end;
					sents[s].rdeps[d.arg_idx] = d;
				}
				else
				{
					dep d; d.head_idx = chunks[i].start - 1; d.arg_idx = chunks[i].end;
					sents[s].rdeps[d.arg_idx] = d;
				}
			}
			else
			{
				if(chunks[i].start == root_idx)
					continue;
				if(chunks[i].start > 0)
				{
					dep d; d.head_idx = chunks[i].start - 1; d.arg_idx = chunks[i].end;
					sents[s].rdeps[d.arg_idx] = d;
				}
				else
				{
					dep d; d.head_idx = root_idx; d.arg_idx = chunks[i].end;
					sents[s].rdeps[d.arg_idx] = d;
				}
			}
		}

		dep d; d.arg_idx = root_idx; d.head_idx = sents[s].words.size() - 1;
		sents[s].rdeps[root_idx] = d;
		int x = 0;
	}
}

void data::write_output()
{
	ofstream outf(config["out_put"].c_str());
	for(int i = 0; i < sents.size(); i++)
	{
		for(int j = 0; j < sents[i].words.size(); j++)
		{
			outf<<j<<":"<<words[sents[i].words[j].word]<<"/"<<gtags[sents[i].words[j].tag]<<"/"<<ctags[sents[i].words[j].ctag]<<" ";
		}
		
		outf<<endl;outf<<"gold: ";
		char head[10]; char arg[10];
		for(int j = 0; j < sents[i].gdeps.size(); j++)
		{
			dep dp = sents[i].gdeps[j];
			itoa(dp.head_idx,head,10);
			itoa(dp.arg_idx,arg,10);
			outf<<head<<"-"<<arg<<" ";
		}
		outf<<endl;outf<<"rule: ";
		for(int j = 0; j < sents[i].rdeps.size(); j++)
		{
			dep dp = sents[i].rdeps[j];
			itoa(dp.head_idx,head,10);
			itoa(dp.arg_idx,arg,10);
			outf<<head<<"-"<<arg<<" ";
		}
		outf<<endl;outf<<"pred: ";
		for(int j = 0; j < sents[i].pdeps.size(); j++)
		{
			dep dp = sents[i].pdeps[j];
			itoa(dp.head_idx,head,10);
			itoa(dp.arg_idx,arg,10);
			outf<<head<<"-"<<arg<<" ";
		}
		outf<<endl<<endl;			
	}
}

void data::print_results(bool directed)
{

	int correct = 0;
	int total = 0;

	for(int i = 0; i < sents.size(); i++)
	{
		int len = sents[i].words.size();
		int pidx = 0;
		for(; pidx < sents[i].pdeps.size(); pidx++)
		{
			dep dp = sents[i].pdeps[pidx];
			int gidx = 0;
			for(; gidx < sents[i].gdeps.size(); gidx++)
				if(directed)
				{
					if(sents[i].gdeps[gidx] == dp)
						break;
				}
				else
				{	
					if( equals(sents[i].gdeps[gidx], dp) )
						break;				
				}
				
			if(gidx != sents[i].gdeps.size())
				correct++;
			total++;
		}
	}

	if(directed)
		cout<< "\tDirected\t" << (double)correct / total <<endl;
	else
		cout<< "\tUndirected\t" << (double)correct / total <<endl;
}

void data::print_rule_results(bool directed)
{

	int correct = 0;
	int total = 0;

	for(int i = 0; i < sents.size(); i++)
	{
		int len = sents[i].words.size();
		int pidx = 0;
		for(; pidx < sents[i].rdeps.size(); pidx++)
		{
			dep dp = sents[i].rdeps[pidx];
			int gidx = 0;
			for(; gidx < sents[i].gdeps.size(); gidx++)
				if(directed)
				{
					if(sents[i].gdeps[gidx] == dp)
						break;
				}
				else
				{	
					if( equals(sents[i].gdeps[gidx], dp) )
						break;				
				}
				
			if(gidx != sents[i].gdeps.size())
				correct++;
			total++;
		}
	}

	if(directed)
		cout<< "\tDirected\t" << (double)correct / total <<endl;
	else
		cout<< "\tUndirected\t" << (double)correct / total <<endl;
}

struct wf{int word; int freq;};
bool operator<(const wf &p1, const wf &p2)
{
	return p1.freq > p2.freq;
}
void data::print_subtags()
{
	wf w; w.freq = 0; w.word = 0;
	vector< vector< vector<wf> > > freq;
	vector<wf> inner_most(wsize,w);
	int T = atoi(config["T"].c_str());
	vector< vector<wf> > inner(T,inner_most);
	freq.resize(tsize,inner);

	for(int s = 0; s < sents.size(); s++)
	{
		for(int w = 0; w < sents[s].words.size(); w++)
		{
			freq[sents[s].words[w].tag][sents[s].words[w].subtag][sents[s].words[w].word].freq++;
			freq[sents[s].words[w].tag][sents[s].words[w].subtag][sents[s].words[w].word].word = sents[s].words[w].word;
		}
	}

	ofstream out("out.tags");
	for(int i = 0; i < tsize; i++)
	{
		out<<gtags[i]<<":"<<endl;
		for(int j = 0; j < T; j++)
		{
			out<<j<<" : ";
			sort(freq[i][j].begin(),freq[i][j].end());
			for(int k = 0; k < 10 && k < freq[i][j].size(); k++)
				out<<words[freq[i][j][k].word]<<"|"<<freq[i][j][k].freq<<" ";
			out<<endl;
		}
	}

}

void data::write_stats(bool directed)
{
	int **dep_matrix_bad;
	int **dep_matrix_good;
	alloc_arr_2D(tsize, tsize, dep_matrix_bad);
	alloc_arr_2D(tsize, tsize, dep_matrix_good);
	for(int i = 0; i < tsize; i++)
		for(int j = 0; j < tsize; j++)
		{
			dep_matrix_good[i][j] = 0;
			dep_matrix_bad[i][j]  = 0;
		}

	for(int i = 0; i < sents.size(); i++)
	{
		int len = sents[i].words.size();
		int gidx = 0;
		for(; gidx < sents[i].gdeps.size(); gidx++)
		{
			dep dp = sents[i].gdeps[gidx];
			int pidx = 0;
			for(; pidx < sents[i].pdeps.size(); pidx++)
				if(directed)
				{
					if( sents[i].pdeps[pidx] == sents[i].gdeps[gidx] )
						break;
				}
				else
				{
					if( equals(sents[i].pdeps[pidx], dp) )
						break;					
				}

			int head = sents[i].words[dp.head_idx].tag;
			int arg  = sents[i].words[dp.arg_idx].tag;

			if(pidx == sents[i].pdeps.size())
				dep_matrix_bad[head][arg]++;
			else
				dep_matrix_good[head][arg]++;
		}
	}

	string type = "";
	if(directed)
		type = "dir_";
	else
		type = "undir_";
	
	string bfname = type + "fstats_bad";
	string gfname = type + "fstats_good";
	ofstream bstats(bfname.c_str());
	ofstream gstats(gfname.c_str());
		
	for(int i = 0; i < tsize; i++)
		for(int j = 0; j < tsize; j++)
		{
			if(dep_matrix_good[i][j] > 0)
				gstats<<gtags[i]<<"->"<<gtags[j]<<"\t"<<dep_matrix_good[i][j]<<endl;
			if(dep_matrix_bad[i][j] > 0)
				bstats<<gtags[i]<<"->"<<gtags[j]<<"\t"<<dep_matrix_bad[i][j]<<endl;
		}

	delete_arr_2D(tsize, dep_matrix_bad);
	delete_arr_2D(tsize, dep_matrix_good);
}

void data::load_config()
{
	ifstream fcon("config");
	char line[500];
	while( !fcon.eof() )
	{
		fcon.getline(line, 500);
		char *field = strtok(line, "\t\n ");
		char *value = strtok(NULL, "\t\n ");
		config[field] = value;
	}
}

void data::load_data()
{
	load_config();
	
	int last_word = 0;
	word_map["#"] = 0;
	words.push_back("#");
		
	senlen = 0;

	ifstream fdep(config["deps"].c_str());
	ifstream ftag(config["poses"].c_str());
	ifstream fwrd(config["words"].c_str());

	char tline[5000];
	char dline[5000];
	char wline[5000];
	
	int sen_count = 0;
	while(!ftag.eof() && sen_count < atoi(config["count"].c_str()))
	{
		sen_count++;

		ftag.getline(tline,5000);
		fdep.getline(dline,5000);
		fwrd.getline(wline,5000);

		if(strcmp(tline,"")==0)
			continue;

		sentence s;

		char * tok = strtok(tline," ");
		while(tok)
		{
			token w;
			if(gtag_map.find(tok) == gtag_map.end())
			{
				int pos     = gtag_map.size();
				gtag_map[tok] = pos;
				gtags.push_back(tok);
				w.tag       = pos;
			}
			else
			{
				int pos     = gtag_map[tok];
				w.tag       = pos;
			}
			s.words.push_back(w);
			tok = strtok(NULL, " ");
		}

		int widx = 0;
		tok = strtok(wline," ");
		while(tok)
		{
			if(word_map.find(tok) == word_map.end())
			{
				int word     	= word_map.size();
				word_map[tok] 	= word;
				words.push_back(tok);
				s.words[widx].word = word;
			}
			else
			{
				int word     		 = word_map[tok];
				s.words[widx].word = word;
			}
			widx++;
			tok = strtok(NULL, " ");
		}

		tok = strtok(dline," ");
		string num;
		while(tok)
		{
			dep d; int i;
			for(i=0;i<strlen(tok);i++)
			{
				if(tok[i] == '-')
				{
					d.head_idx = atoi(num.c_str());
					num.clear();
				}
				else
					num += tok[i];
			}

			d.arg_idx = atoi(num.c_str());
			num.clear();

			s.gdeps.push_back(d);
			tok = strtok(NULL, " ");
		}

		sents.push_back(s);
		if(s.words.size() > senlen)
			senlen = s.words.size();
	}

	tsize = gtags.size();
	wsize = words.size();
	
	populate_rule_deps();

	load_tagset_map();
	load_rules();
	load_dir_rules();
	load_neg_rules();
}

struct back_info
{
	double prob;
	int argidx;
	int argstag;
	int split;
	int kids;
};
class model
{
	data d;

	int T;							//truncation level
	int tsize;						//observed tag set size
	int slen;						//max sentence length
	int wsize;						//lexicon size (the size of the emission dirichlet)

	//hyper parameters
	double alpha_0;					//parameters of the GEM prior over top level subsymbol weights i.e. betas
	double alpha_1;					//concentration parameter for the DPs over subtags

	double ems_hyp;					//parameter of a symmetric dirichlet prior for all emission multinomials
	double stp_hyp;					//parameter of a symmetric dirichlet prior for all stop/continue binomial
	double tag_hyp;					//parameter of a symmetric dirichlet prior for all tag multinomials

	//top level weights for all hdp's (there is one hdp per tag)
	double **betas;					//[tag][subtag]   (truncated GEM)

	//probability array
	double         *****stop_prob;  //[Head_tag][Head_subtag][Dir][Kids][isStop]
	double   *****choose_tag_prob;  //[Head_tag][Head_subtag][Dir][Kids][Arg_tag]
	double ******choose_stag_prob;  //[Head_tag][Head_subtag][Dir][Kids][Arg_tag][Arg_subtag]  (truncated DP)
	double	         ***emissions;  //[tag][subtag][word]

	//arrays to hold expected counts
	double			****stop_LHS;	//[Head_tag][Head_subtag][Dir][Kids]
	double		   *****stop_RHS;   //[Head_tag][Head_subtag][Dir][Kids][isStop]
	double    ****choose_tag_LHS;	//[Head_tag][Head_subtag][Dir][Kids]
	double   *****choose_tag_RHS;	//[Head_tag][Head_subtag][Dir][Kids][Arg_tag]
	double  *****choose_stag_LHS;	//[Head_tag][Head_subtag][Dir][Kids][Arg_tag]
	double ******choose_stag_RHS;	//[Head_tag][Head_subtag][Dir][Kids][Arg_tag][Arg_subtag]
	double 				  **tags;	//[tag][subtag]
	double		    ***tag_words;	//[tag][subtag][word]

	//partition function
	double partition;

	//charts for inside-outside
	double ******beta;				//[head_index][head_subtag][Seal type][Kids][i][j]
	double ******alpha;				//[head_index][head_subtag][Seal type][Kids][i][j]

	//chart for cky
	back_info ******back;		//[head_index][head_subtag][Seal type][Kids][i][j]

	int noun;
	
	//for constraint function
	typedef bool (model::*fptr)(int, int, Kids, sentence &);
	vector<double> lembda;
	vector<double> constr_ecount;
	vector<double> constr_bound;
	vector<double> gradient;
	vector<fptr> constr_func;

	void init_probs();
	void init_zero_counts();
	void init_harmonic_counts();
	void alloc_mem();

	void Inside(sentence &s);
	void Outside(sentence &s);
	void E_Step();
	void var_M_Step();
	void M_Step();
	void estimate_betas();
	double compute_beta_gradient(int tag, int subtag);
	double compute_beta_objective(int tag);
	void CKY(sentence &s);
	void get_pdeps(int hidx, int hstag, Seal head_seal_type, Kids kids, int i, int j, vector<dep> &deps, vector<token> &words);

	void var_bound();
	
	void gradient_search();
	void compute_gradient();
	void gradient_E_Step();

	bool rule_constr_fulfilled(int h_idx, int a_idx, Kids kids, sentence &s);
	bool chunk_constr_fulfilled(int h_idx, int a_idx, Kids kids, sentence &s);
	bool neg_rule_constr_fulfilled(int h_idx, int a_idx, Kids kids, sentence &s);
	
	double get_constr_value(int &h_idx, int &a_idx, Kids kids, sentence &s);
	double get_gen_prob(int &htag, int &hstag, Dir dir, Kids kids, int &atag, int &astag);
	void update_gen_ecounts(int &htag, int &hstag, Dir dir, Kids kids, int &atag, int &astag, double &prob);

	void print_betas();
	void print_subtags();
public:
	model();
	~model();
	void learn();
	void Annotate();
	void output();
};

struct wp{int word; double prob;};
bool operator<(const wp &p1, const wp &p2)
{
	return p1.prob > p2.prob;
}
void model::print_subtags()
{
	ofstream out("tags.out");
	wp x; x.prob=0; x.word=0;
	vector<wp> wpv(wsize,x);
	for(int tag = 0; tag < tsize; tag++)
	{
		out<<d.gtags[tag]<<":"<<endl;
		for(int stag = 0; stag < T; stag++)
		{
			out<<stag<<" : ";
			for(int w = 0; w < wsize; w++)
			{
				wpv[w].word = w;
				wpv[w].prob = emissions[tag][stag][w];
			}
			sort(wpv.begin(),wpv.end());
			for(int k = 0; k < 10 && k < wpv.size(); k++)
				out<<d.words[wpv[k].word]<<"|"<<wpv[k].prob<<" ";
			out<<endl;
		}
	}
}

double model::get_gen_prob(int &htag, int &hstag, Dir dir, Kids kids, int &atag, int &astag)
{
	return (choose_tag_prob[htag][hstag][dir][kids][atag] *
			choose_stag_prob[htag][hstag][dir][kids][atag][astag] *
			stop_prob[htag][hstag][dir][kids][CONTINUE]);
}

void model::alloc_mem()
{
	//top level weights for all HDPs
	alloc_arr_2D(tsize, T, betas);

	//probability array
	alloc_arr_5D(tsize, T, 2, 3, 2, stop_prob);  //[Head_tag][Head_subtag][Dir][Kids][isStop]
	alloc_arr_5D(tsize, T, 2, 3, tsize, choose_tag_prob);  //[Head_tag][Head_subtag][Dir][Kids][Arg_tag]
	alloc_arr_6D(tsize, T, 2, 3, tsize, T, choose_stag_prob);  //[Head_tag][Head_subtag][Dir][Kids][Arg_tag][Arg_subtag]  (truncated DP)
	alloc_arr_3D(tsize, T, wsize, emissions);  //[tag][subtag][word]

	//arrays to hold expected counts
	alloc_arr_4D(tsize, T, 2, 3, stop_LHS);	//[Head_tag][Head_subtag][Dir][Kids]
	alloc_arr_5D(tsize, T, 2, 3, 2, stop_RHS);   //[Head_tag][Head_subtag][Dir][Kids][isStop]
	alloc_arr_4D(tsize, T, 2, 3, choose_tag_LHS);	//[Head_tag][Head_subtag][Dir][Kids]
	alloc_arr_5D(tsize, T, 2, 3, tsize, choose_tag_RHS);	//[Head_tag][Head_subtag][Dir][Kids][Arg_tag]
	alloc_arr_5D(tsize, T, 2, 3, tsize, choose_stag_LHS);	//[Head_tag][Head_subtag][Dir][Kids][Arg_tag]
	alloc_arr_6D(tsize, T, 2, 3, tsize, T, choose_stag_RHS);	//[Head_tag][Head_subtag][Dir][Kids][Arg_tag][Arg_subtag]
	alloc_arr_2D(tsize, T, tags);	//[tag][subtag]
	alloc_arr_3D(tsize, T, wsize, tag_words);	//[tag][subtag][word]

	//charts for inside-outside
	alloc_arr_6D(slen, T, 3, 3, slen, slen, beta);				//[head_index][head_subtag][Seal type][Kids][i][j]
	alloc_arr_6D(slen, T, 3, 3, slen, slen, alpha);				//[head_index][head_subtag][Seal type][Kids][i][j]

	//chart for cky
	alloc_arr_6D(slen, T, 3, 3, slen, slen, back);
}

model::~model()
{
	//top level weights for all HDPs
	delete_arr_2D(tsize, betas);

	//probability array
	delete_arr_5D(tsize, T, 2, 3, stop_prob);  //[Head_tag][Head_subtag][Dir][Kids][isStop]
	delete_arr_5D(tsize, T, 2, 3, choose_tag_prob);  //[Head_tag][Head_subtag][Dir][Kids][Arg_tag]
	delete_arr_6D(tsize, T, 2, 3, tsize, choose_stag_prob);  //[Head_tag][Head_subtag][Dir][Kids][Arg_tag][Arg_subtag]  (truncated DP)
	delete_arr_3D(tsize, T, emissions);  //[tag][subtag][word]

	//arrays to hold expected counts
	delete_arr_4D(tsize, T, 2, stop_LHS);	//[Head_tag][Head_subtag][Dir][Kids]
	delete_arr_5D(tsize, T, 2, 3, stop_RHS);   //[Head_tag][Head_subtag][Dir][Kids][isStop]
	delete_arr_4D(tsize, T, 2, choose_tag_LHS);	//[Head_tag][Head_subtag][Dir][Kids]
	delete_arr_5D(tsize, T, 2, 3, choose_tag_RHS);	//[Head_tag][Head_subtag][Dir][Kids][Arg_tag]
	delete_arr_5D(tsize, T, 2, 3, choose_stag_LHS);	//[Head_tag][Head_subtag][Dir][Kids][Arg_tag]
	delete_arr_6D(tsize, T, 2, 3, tsize, choose_stag_RHS);	//[Head_tag][Head_subtag][Dir][Kids][Arg_tag][Arg_subtag]
	delete_arr_2D(tsize, tags);	//[tag][subtag]
	delete_arr_3D(tsize, T, tag_words);	//[tag][subtag][word]

	//charts for inside-outside
	delete_arr_6D(slen, T, 3, 3, slen, beta);				//[head_index][head_subtag][Seal type][Kids][i][j]
	delete_arr_6D(slen, T, 3, 3, slen, alpha);				//[head_index][head_subtag][Seal type][Kids][i][j]

	//chart for cky
	delete_arr_6D(slen, T, 3, 3, slen, back);
}
model::model()
{
	d.load_data();
	cout<<"loaded data\n";

	T     = atoi(d.config["T"].c_str());
	tsize = d.tsize;
	wsize = d.wsize;
	slen  = d.senlen;
	alpha_0 = atof(d.config["alpha_0"].c_str());
	alpha_1 = atof(d.config["alpha_1"].c_str());	
	ems_hyp = atof(d.config["hyp"].c_str());
	stp_hyp = atof(d.config["hyp"].c_str());
	tag_hyp = atof(d.config["hyp"].c_str());

	alloc_mem();
	cout<<"allocated_memory\n";
	for(int t = 0; t < tsize; t++)
		for(int st = 0; st < T; st++)
			betas[t][st] = 1.0 / T;
	
	//alternative initialization
	//init_harmonic_counts();
	//var_M_Step();
	noun = d.ctag_map["Noun"];

	init_probs();
	cout<<"initialized probabilities\n";
	
	cout<<"rule dependencies' accuracy\n";
	d.print_rule_results(true);
	d.print_rule_results(false);
	
	if ( constr_count > 0 )
	{
		constr_func.resize(constr_count);
		constr_func[RULE]  = &model::rule_constr_fulfilled;
	
		constr_ecount.resize(constr_count);
		constr_bound.resize(constr_count);
		gradient.resize(constr_count);
		lembda.resize(constr_count);


		double total = 0;
		for(int s = 0; s < d.sents.size(); s++)
			total += (d.sents[s].words.size() - 1);	
		constr_bound[RULE]     = total * (1.0 - atof(d.config["threshold"].c_str()) );

		lembda[RULE]   = 0;
	}
}

bool model::rule_constr_fulfilled(int h_idx, int a_idx, Kids kids, sentence &s)
{
	//return !(a_idx < s.rdeps.size() && h_idx == s.rdeps[a_idx].head_idx);

	//return !(a_idx < s.gdeps.size() && h_idx == s.gdeps[a_idx].head_idx);
	
	return !d.rules[s.words[h_idx].ctag][s.words[a_idx].ctag];
	
	//Dir dir = h_idx < a_idx ? RIGHT : LEFT ;
	//return !d.dir_rules[dir][s.words[h_idx].tag][s.words[a_idx].tag];
	
	//return !((a_idx < s.rdeps.size() && h_idx == s.rdeps[a_idx].head_idx) || d.rules[s.words[h_idx].ctag][s.words[a_idx].ctag]);
}

bool model::chunk_constr_fulfilled(int h_idx, int a_idx, Kids kids, sentence &s)
{
	if(s.words[a_idx].ctag == noun && s.words[a_idx+1].ctag == noun)
	{
		int i = a_idx + 1;
		while(s.words[i].ctag == noun) i++;
		if(h_idx == i - 1)
			return false;
		return true;
	}
	else
		return false;
}

bool model::neg_rule_constr_fulfilled(int h_idx, int a_idx, Kids kids, sentence &s)
{
	return !d.neg_rules[s.words[h_idx].ctag][s.words[a_idx].ctag];
}

double model::get_constr_value(int &h_idx, int &a_idx, Kids kids, sentence &s)
{
	double inner_prod = 0;
	for(int i = 0; i < constr_count; i++)
	{
		inner_prod += (this->*constr_func[i])(h_idx, a_idx, kids, s) * (-lembda[i]);
	}
	return exp( inner_prod ) ;
}
void model::init_harmonic_counts()
{
	init_zero_counts();
	srand(time(0));
    int C = 1;
	vector<double> harmonic_sums(d.senlen);
    harmonic_sums[0] = 0;
    for(int i = 1; i < d.senlen; i++)
    {
		harmonic_sums[i] = 1.0/i + C + harmonic_sums[i-1];
    }

	for(int s = 0; s < d.sents.size(); s++)
	{
		for(int hidx = 0; hidx < d.sents[s].words.size(); hidx++)
		{
			int htag = d.sents[s].words[hidx].tag;
			for(int hstag = 0; hstag < T; hstag++)
			{
				for(int kids = NO_KIDS; kids <= TWO_OR_MORE; kids++)
				{				
					for(int aidx = 0; aidx < d.sents[s].words.size(); aidx++)
					{
						int atag = d.sents[s].words[aidx].tag;
						Dir dir  = hidx < aidx ? RIGHT : LEFT ;
						stop_LHS[htag][hstag][dir][kids] +=2;
						stop_RHS[htag][hstag][dir][kids][STOP] +=1;
						stop_RHS[htag][hstag][dir][kids][CONTINUE] +=1;
						double choose_prob = 0;
						if(hidx != aidx)
						{
							int senlen = d.sents[s].words.size();
							if(hidx == (senlen - 1) )
									choose_prob = 1.0/(senlen - 1);
							else if(aidx == (senlen - 1) )
									choose_prob = 0.0;
							else
							{
									senlen--;
									int dist = abs(hidx-aidx);
									float denom = harmonic_sums[senlen - aidx - 1] + harmonic_sums[aidx];
									float num = 1.0/dist + C;
									choose_prob = num/denom;
							}

						}
						choose_tag_LHS[htag][hstag][dir][kids] += choose_prob;
						choose_tag_RHS[htag][hstag][dir][kids][atag] += choose_prob;
						for(int astag = 0; astag < T; astag++)
						{
							double sub_prob = 1.0 + 1.0 / (rand() % 1000 + 10);
							choose_stag_LHS[htag][hstag][dir][kids][atag] += sub_prob;
							choose_stag_RHS[htag][hstag][dir][kids][atag][astag] += sub_prob;
						}
					}
				}
				int w = d.sents[s].words[hidx].word;
				tags[htag][hstag] += 1;
				tag_words[htag][hstag][w] += 1;
			}
		}
	}
}
void model::init_zero_counts()
{
	for(int htag = 0; htag < tsize; htag++)
	{
		for(int hstag = 0; hstag < T; hstag++)
		{
			for(int dir = LEFT; dir <= RIGHT; dir++)
			{
				for(int kids = NO_KIDS; kids <= TWO_OR_MORE; kids++)
				{
					stop_LHS[htag][hstag][dir][kids] = 0;
					for(int stype = STOP; stype <= CONTINUE; stype++)
						stop_RHS[htag][hstag][dir][kids][stype] = 0;
					
					choose_tag_LHS[htag][hstag][dir][kids] = 0;
					for(int atag = 0; atag < tsize; atag++)
					{
						choose_tag_RHS[htag][hstag][dir][kids][atag] = 0;
						choose_stag_LHS[htag][hstag][dir][kids][atag] = 0;
						for(int astag = 0; astag < T; astag++)
							choose_stag_RHS[htag][hstag][dir][kids][atag][astag] = 0;
					}
				}
			}
		
			tags[htag][hstag] = 0;
			for(int w = 0; w < wsize; w++)
				tag_words[htag][hstag][w] = 0;
		}
	}
}
void model::init_probs()
{
	double uniform_stop = 0.5;
	double uniform_tag  = 1.0 / tsize;
	double uniform_stag = 1.0 / T;
	double uniform_word = 1.0 / wsize;
	srand(time(0));

	for(int htag = 0; htag < tsize; htag++)
	{
		for(int hstag = 0; hstag < T; hstag++)
		{
			for(int dir = LEFT; dir <= RIGHT; dir++)
			{
				for(int kids = NO_KIDS; kids <= TWO_OR_MORE; kids++)
				{
					double sum = 0;
					for(int stype = STOP; stype <= CONTINUE; stype++)
					{
						stop_prob[htag][hstag][dir][kids][stype] = uniform_stop;// + 1.0 / (rand() %1000 + 100);
						sum += stop_prob[htag][hstag][dir][kids][stype];
					}
					for(int stype = STOP; stype <= CONTINUE; stype++)
						stop_prob[htag][hstag][dir][kids][stype] = stop_prob[htag][hstag][dir][kids][stype] / sum;

					sum = 0;
					for(int atag = 0; atag < tsize; atag++)
					{
						choose_tag_prob[htag][hstag][dir][kids][atag] = uniform_tag;// + 1.0 / (rand() %1000 + 100);
						sum += choose_tag_prob[htag][hstag][dir][kids][atag];
					}
					for(int atag = 0; atag < tsize; atag++)
						choose_tag_prob[htag][hstag][dir][kids][atag] = choose_tag_prob[htag][hstag][dir][kids][atag] / sum;

					for(int atag = 0; atag < tsize; atag++)
					{
						sum = 0;
						for(int astag = 0; astag < T; astag++)
						{
							choose_stag_prob[htag][hstag][dir][kids][atag][astag] = uniform_stag + 1.0 / (rand() %1000 + 100);
							sum += choose_stag_prob[htag][hstag][dir][kids][atag][astag];
						}
						for(int astag = 0; astag < T; astag++)
							choose_stag_prob[htag][hstag][dir][kids][atag][astag] = choose_stag_prob[htag][hstag][dir][kids][atag][astag] / sum;
					}
				}
			}

			double sum = 0;
			for(int w = 0; w < wsize; w++)
			{
				emissions[htag][hstag][w] = uniform_word;//  + 1.0 / (rand() %10000 + 100);
				sum += emissions[htag][hstag][w];
			}
			for(int w = 0; w < wsize; w++)
			{
				emissions[htag][hstag][w] = emissions[htag][hstag][w] / sum;
			}
		}
	}
}

void model::Inside(sentence &s)
{
	int senlen = s.words.size();
	
	//initialize
	for(int n = 0; n < senlen; n++)
		for(int stype = R; stype <= F; stype++)
			for(int i = 0; i < senlen; i++)
				for(int j = 0; j < senlen; j++)
					for(int stag = 0; stag < T; stag++)
						for(int kids = NO_KIDS; kids <= TWO_OR_MORE; kids++)
							beta[n][stag][stype][kids][i][j]   = 0;

	//base case
	for(int n = 0; n < senlen; n++)
	{
		int word = s.words[n].word;
		int tag  = s.words[n].tag;
		for(int stag = 0; stag < T; stag++)
		{
			beta[n][stag][L][NO_KIDS][n][n] = emissions[tag][stag][word];
			beta[n][stag][R][NO_KIDS][n][n] = emissions[tag][stag][word];
			//beta[n][stag][F][NO_KIDS][n][n] = beta[n][stag][L][NO_KIDS][n][n] * beta[n][stag][R][NO_KIDS][n][n] / emissions[tag][stag][word];
		}
	}

	//recurse
	for(int size = 1; size < senlen; size++)
		for(int i = 0; i < senlen - size ; i++)
			{
				int j = i + size - 1;
				for(int k = i; k < j; k++)
				{
					int hidx = i;
					int htag = s.words[hidx].tag;
					for(int r = k+1; r <=j; r++)
					{
						int aidx = r;
						int atag = s.words[aidx].tag;
						for(int hstag = 0; hstag < T; hstag++)
						{
							for(int astag = 0; astag < T; astag++)
							{
								beta[hidx][hstag][R][ONE_KID][i][j]
								+= get_gen_prob(htag, hstag, RIGHT, NO_KIDS, atag, astag) *
									get_constr_value(hidx, aidx, NO_KIDS, s) *
									beta[hidx][hstag][R][NO_KIDS][i][k] *
									beta[aidx][astag][F][NO_KIDS][k+1][j];
							
								beta[hidx][hstag][R][TWO_OR_MORE][i][j]
								+= get_gen_prob(htag, hstag, RIGHT, ONE_KID, atag, astag) *
									get_constr_value(hidx, aidx, ONE_KID, s) *
									beta[hidx][hstag][R][ONE_KID][i][k] *
									beta[aidx][astag][F][NO_KIDS][k+1][j];

								beta[hidx][hstag][R][TWO_OR_MORE][i][j]
								+= get_gen_prob(htag, hstag, RIGHT, TWO_OR_MORE, atag, astag) *
									get_constr_value(hidx, aidx, TWO_OR_MORE, s) *
									beta[hidx][hstag][R][TWO_OR_MORE][i][k] *
									beta[aidx][astag][F][NO_KIDS][k+1][j];
							}
						}
					}

					hidx = j;
					htag = s.words[hidx].tag;
					for(int l = i; l <= k; l++)
					{
						int aidx = l;
						int atag = s.words[aidx].tag;
						for(int hstag = 0; hstag < T; hstag++)
						{
							for(int astag = 0; astag < T; astag++)
							{
								beta[hidx][hstag][L][ONE_KID][i][j]
								+= get_gen_prob(htag, hstag, LEFT, NO_KIDS, atag, astag) *
									get_constr_value(hidx, aidx, NO_KIDS, s) *
									beta[hidx][hstag][L][NO_KIDS][k+1][j] *
									beta[aidx][astag][F][NO_KIDS][i][k];
							
								beta[hidx][hstag][L][TWO_OR_MORE][i][j]
								+= get_gen_prob(htag, hstag, LEFT, ONE_KID, atag, astag) *
									get_constr_value(hidx, aidx, ONE_KID, s) *
									beta[hidx][hstag][L][ONE_KID][k+1][j] *
									beta[aidx][astag][F][NO_KIDS][i][k];

								beta[hidx][hstag][L][TWO_OR_MORE][i][j]
								+= get_gen_prob(htag, hstag, LEFT, TWO_OR_MORE, atag, astag) *
									get_constr_value(hidx, aidx, TWO_OR_MORE, s) *
									beta[hidx][hstag][L][TWO_OR_MORE][k+1][j] *
									beta[aidx][astag][F][NO_KIDS][i][k];
							}
						}
					}
				}
				
				//--------------------------------------------------------------------------------
				for(int hidx = i; hidx <= j; hidx++)
				{
					int htag = s.words[hidx].tag;
					int word = s.words[hidx].word;
					for(int hstag = 0; hstag < T; hstag++)
						for(int lkids = NO_KIDS; lkids <=TWO_OR_MORE; lkids++)
							for(int rkids = NO_KIDS; rkids <=TWO_OR_MORE; rkids++)
								beta[hidx][hstag][F][NO_KIDS][i][j] +=
									stop_prob[htag][hstag][RIGHT][rkids][STOP] *
									beta[hidx][hstag][R][rkids][hidx][j] *
									stop_prob[htag][hstag][LEFT][lkids][STOP] *
									beta[hidx][hstag][L][lkids][i][hidx] /
									emissions[htag][hstag][word];
				}
			}

	//Dealing with the root symbol
	int hidx = senlen - 1;
	int htag = s.words[hidx].tag;
	int hstag = 0 ; //root tag has only one subtag	
	int i = 0; 
	int j = hidx;
	int k = j - 1;
	for(int aidx = 0; aidx < hidx; aidx++)
	{
		int atag = s.words[aidx].tag;
		for(int astag = 0; astag < T; astag++)
		{
			beta[hidx][hstag][L][ONE_KID][i][j]
			+= get_gen_prob(htag, hstag, LEFT, NO_KIDS, atag, astag) *
			   get_constr_value(hidx, aidx, NO_KIDS, s) *
			   beta[hidx][hstag][L][NO_KIDS][k+1][j] *
			   beta[aidx][astag][F][NO_KIDS][i][k];
		}
	}

	int word = s.words[hidx].word;
	beta[hidx][hstag][F][NO_KIDS][i][j]
	+= stop_prob[htag][hstag][RIGHT][NO_KIDS][STOP] *
		beta[hidx][hstag][R][NO_KIDS][j][j] *
		stop_prob[htag][hstag][LEFT][ONE_KID][STOP] *
	    beta[hidx][hstag][L][ONE_KID][i][j] /
		emissions[htag][hstag][word];

}

void model::Outside(sentence &s)
{
	int senlen = s.words.size();
	
	//initialize
	for(int n = 0; n < senlen; n++)
		for(int stype = R; stype <= F; stype++)
			for(int i = 0; i < senlen; i++)
				for(int j = 0; j < senlen; j++)
					for(int stag = 0; stag < T; stag++)
						for(int kids = NO_KIDS; kids <= TWO_OR_MORE; kids++)
							alpha[n][stag][stype][kids][i][j]   = 0;

	//base case
	int i = 0; int j = senlen - 1; int hidx = senlen - 1; 
	int htag  = s.words[hidx].tag;
	int word  = s.words[hidx].word;
	int hstag = 0;
	alpha[hidx][hstag][F][NO_KIDS][i][j]  = 1;
	alpha[hidx][hstag][L][ONE_KID][i][j]  = alpha[hidx][hstag][F][NO_KIDS][i][j] *
										stop_prob[htag][hstag][LEFT][ONE_KID][STOP] *
										beta[hidx][hstag][R][NO_KIDS][j][j] *
										stop_prob[htag][hstag][RIGHT][NO_KIDS][STOP] /
										emissions[htag][hstag][word];
	alpha[hidx][hstag][R][NO_KIDS][j][j]  = alpha[hidx][hstag][F][NO_KIDS][i][j] *
										stop_prob[htag][hstag][LEFT][ONE_KID][STOP] *
										beta[hidx][hstag][L][ONE_KID][i][j] *
										stop_prob[htag][hstag][RIGHT][NO_KIDS][STOP] /
										emissions[htag][hstag][word];

	int k = senlen - 1; j = senlen - 2; i = 0;
	for(int aidx = i; aidx <= j; aidx++)
	{
		int atag  = s.words[aidx].tag;
		for(int astag = 0; astag < T; astag++)
		{
			alpha[aidx][astag][F][NO_KIDS][i][j]  
			+= beta[hidx][hstag][L][NO_KIDS][j+1][k] *
			   alpha[hidx][hstag][L][ONE_KID][i][k] *
			   get_gen_prob(htag, hstag, LEFT, NO_KIDS, atag, astag) *
			   get_constr_value(hidx, aidx, NO_KIDS, s);
		}
	}

	//recurse
	for(int size = senlen - 1; size >= 1; size--)
		for(int i = 0; i < (senlen - size); i++)
			{
				int j = i + size - 1;
				for(int k = 0; k < i; k++)
				{
					int hidx = k;
					for(int aidx = i; aidx <= j; aidx++)
					{
						int htag = s.words[hidx].tag;
						int atag = s.words[aidx].tag;
						for(int hstag = 0; hstag < T; hstag++)
						{
							for(int astag = 0; astag < T; astag++)
							{
									alpha[aidx][astag][F][NO_KIDS][i][j]  
									+= beta[hidx][hstag][R][NO_KIDS][k][i-1] *
									   alpha[hidx][hstag][R][ONE_KID][k][j] *
									   get_gen_prob(htag, hstag, RIGHT, NO_KIDS, atag, astag) *
									   get_constr_value(hidx, aidx, NO_KIDS, s);

									alpha[aidx][astag][F][NO_KIDS][i][j]  
									+= beta[hidx][hstag][R][ONE_KID][k][i-1] *
									   alpha[hidx][hstag][R][TWO_OR_MORE][k][j] *
									   get_gen_prob(htag, hstag, RIGHT, ONE_KID, atag, astag) *
									   get_constr_value(hidx, aidx, ONE_KID, s);

									alpha[aidx][astag][F][NO_KIDS][i][j]  
									+= beta[hidx][hstag][R][TWO_OR_MORE][k][i-1] *
									   alpha[hidx][hstag][R][TWO_OR_MORE][k][j] *
									   get_gen_prob(htag, hstag, RIGHT, TWO_OR_MORE, atag, astag) *
									   get_constr_value(hidx, aidx, TWO_OR_MORE, s);
							}
						}
					}

					hidx = j;
					for(int aidx = k; aidx < i; aidx++)
					{
						int htag = s.words[hidx].tag;
						int atag = s.words[aidx].tag;
						for(int hstag = 0; hstag < T; hstag++)
						{
							for(int astag = 0; astag < T; astag++)
							{
									alpha[hidx][hstag][L][NO_KIDS][i][j]  
									+= beta[aidx][astag][F][NO_KIDS][k][i-1] *
									   alpha[hidx][hstag][L][ONE_KID][k][j] *
									   get_gen_prob(htag, hstag, LEFT, NO_KIDS, atag, astag) *
									   get_constr_value(hidx, aidx, NO_KIDS, s);

									alpha[hidx][hstag][L][ONE_KID][i][j]  
									+= beta[aidx][astag][F][NO_KIDS][k][i-1] *
									   alpha[hidx][hstag][L][TWO_OR_MORE][k][j] *
									   get_gen_prob(htag, hstag, LEFT, ONE_KID, atag, astag) *
									   get_constr_value(hidx, aidx, ONE_KID, s);

									alpha[hidx][hstag][L][TWO_OR_MORE][i][j]  
									+= beta[aidx][astag][F][NO_KIDS][k][i-1] *
									   alpha[hidx][hstag][L][TWO_OR_MORE][k][j] *
									   get_gen_prob(htag, hstag, LEFT, TWO_OR_MORE, atag, astag) *
									   get_constr_value(hidx, aidx, TWO_OR_MORE, s);
							}
						}
					}
				}
				for(int k = j + 1; k < senlen - 1; k++)
				{
					int hidx = k;
					for(int aidx = i; aidx <= j; aidx++)
					{
						int htag = s.words[hidx].tag;
						int atag = s.words[aidx].tag;
						for(int hstag = 0; hstag < T; hstag++)
						{
							for(int astag = 0; astag < T; astag++)
							{
									alpha[aidx][astag][F][NO_KIDS][i][j]  
									+= beta[hidx][hstag][L][NO_KIDS][j+1][k] *
									   alpha[hidx][hstag][L][ONE_KID][i][k] *
									   get_gen_prob(htag, hstag, LEFT, NO_KIDS, atag, astag) *
									   get_constr_value(hidx, aidx, NO_KIDS, s);

									alpha[aidx][astag][F][NO_KIDS][i][j]  
									+= beta[hidx][hstag][L][ONE_KID][j+1][k] *
									   alpha[hidx][hstag][L][TWO_OR_MORE][i][k] *
									   get_gen_prob(htag, hstag, LEFT, ONE_KID, atag, astag) *
									   get_constr_value(hidx, aidx, ONE_KID, s);

									alpha[aidx][astag][F][NO_KIDS][i][j]  
									+= beta[hidx][hstag][L][TWO_OR_MORE][j+1][k] *
									   alpha[hidx][hstag][L][TWO_OR_MORE][i][k] *
									   get_gen_prob(htag, hstag, LEFT, TWO_OR_MORE, atag, astag) *
									   get_constr_value(hidx, aidx, TWO_OR_MORE, s);
							}
						}
					}

					hidx = i;
					for(int aidx = j + 1; aidx <= k; aidx++)
					{
						int htag = s.words[hidx].tag;
						int atag = s.words[aidx].tag;
						for(int hstag = 0; hstag < T; hstag++)
						{
							for(int astag = 0; astag < T; astag++)
							{
									alpha[hidx][hstag][R][NO_KIDS][i][j]  
									+= beta[aidx][astag][F][NO_KIDS][j+1][k] *
									   alpha[hidx][hstag][R][ONE_KID][i][k] *
									   get_gen_prob(htag, hstag, RIGHT, NO_KIDS, atag, astag) *
									   get_constr_value(hidx, aidx, NO_KIDS, s);

									alpha[hidx][hstag][R][ONE_KID][i][j]  
									+= beta[aidx][astag][F][NO_KIDS][j+1][k] *
									   alpha[hidx][hstag][R][TWO_OR_MORE][i][k] *
									   get_gen_prob(htag, hstag, RIGHT, ONE_KID, atag, astag) *
									   get_constr_value(hidx, aidx, ONE_KID, s);

									alpha[hidx][hstag][R][TWO_OR_MORE][i][j]  
									+= beta[aidx][astag][F][NO_KIDS][j+1][k] *
									   alpha[hidx][hstag][R][TWO_OR_MORE][i][k] *
									   get_gen_prob(htag, hstag, RIGHT, TWO_OR_MORE, atag, astag) *
									   get_constr_value(hidx, aidx, TWO_OR_MORE, s);
							}
						}
					}
				}
				int tag = s.words[i].tag;
				int word = s.words[i].word;
				for(int k = 0; k <= i; k++)
				{
					for(int lkids = NO_KIDS; lkids <= TWO_OR_MORE; lkids++)
						for(int rkids = NO_KIDS; rkids <= TWO_OR_MORE; rkids++)
							for(int stag = 0; stag < T; stag++)
								alpha[i][stag][R][rkids][i][j] +=
									alpha[i][stag][F][NO_KIDS][k][j] *
									stop_prob[tag][stag][RIGHT][rkids][STOP] *
									stop_prob[tag][stag][LEFT][lkids][STOP] *
									beta[i][stag][L][lkids][k][i] /
									emissions[tag][stag][word];

				}
				tag = s.words[j].tag;
				word = s.words[j].word;
				for(int k = j; k < senlen - 1; k++)
				{
					for(int lkids = NO_KIDS; lkids <= TWO_OR_MORE; lkids++)
						for(int rkids = NO_KIDS; rkids <= TWO_OR_MORE; rkids++)
							for(int stag = 0; stag < T; stag++)
								alpha[j][stag][L][lkids][i][j] +=
									alpha[j][stag][F][NO_KIDS][i][k] *
									stop_prob[tag][stag][RIGHT][rkids][STOP] *
									stop_prob[tag][stag][LEFT][lkids][STOP] *
									beta[j][stag][R][rkids][j][k] /
									emissions[tag][stag][word];
				}
			}
}

void model::update_gen_ecounts(int &htag, int &hstag, Dir dir, Kids kids, int &atag, int &astag, double &prob)
{
	choose_tag_LHS[htag][hstag][dir][kids] += prob;
	choose_tag_RHS[htag][hstag][dir][kids][atag] += prob;
	choose_stag_LHS[htag][hstag][dir][kids][atag] += prob;
	choose_stag_RHS[htag][hstag][dir][kids][atag][astag] += prob;
	stop_LHS[htag][hstag][dir][kids] += prob;
	stop_RHS[htag][hstag][dir][kids][CONTINUE] += prob;
}
void model::E_Step()
{
	//initialize
	init_zero_counts();
	partition = 0;
		
	for(int sidx = 0; sidx < d.sents.size(); sidx++)
	{
		sentence s = d.sents[sidx];
		Inside(s);
		Outside(s);

		int senlen = s.words.size();
		double sen_prob = beta[senlen-1][0][F][NO_KIDS][0][senlen-1];
		double prob, num;
		partition += log(sen_prob);

		if(senlen == 3)
			int x = 0;
		for(int n = 0; n < senlen; n++)
		{
			int word = s.words[n].word;
			int tag  = s.words[n].tag;
			for(int stag = 0; stag < T; stag++)
			{
				num = alpha[n][stag][R][NO_KIDS][n][n] * beta[n][stag][R][NO_KIDS][n][n];
				prob = num / sen_prob;
				tags[tag][stag] += prob;
				tag_words[tag][stag][word] += prob;
			}
		}

		for(int size = 1; size <= senlen; size++)
			for(int i = 0; i <= senlen - size ; i++)
				{
					int j = i + size - 1;
					for(int k = i; k < j; k++)
					{
						int hidx = i;
						int htag = s.words[hidx].tag;
						for(int r = k+1; r <=j; r++)
						{
							int aidx = r;
							int atag = s.words[aidx].tag;
							for(int hstag = 0; hstag < T; hstag++)
							{
								for(int astag = 0; astag < T; astag++)
								{
										num = alpha[hidx][hstag][R][ONE_KID][i][j] *
													 get_gen_prob(htag, hstag, RIGHT, NO_KIDS, atag, astag) *
													 get_constr_value(hidx, aidx, NO_KIDS, s) *
													 beta[hidx][hstag][R][NO_KIDS][i][k] *
													 beta[aidx][astag][F][NO_KIDS][k+1][j];
										prob = num / sen_prob;
										update_gen_ecounts(htag, hstag, RIGHT, NO_KIDS, atag, astag, prob);

										num = alpha[hidx][hstag][R][TWO_OR_MORE][i][j] *
													 get_gen_prob(htag, hstag, RIGHT, ONE_KID, atag, astag) *
													 get_constr_value(hidx, aidx, ONE_KID, s) *
													 beta[hidx][hstag][R][ONE_KID][i][k] *
													 beta[aidx][astag][F][NO_KIDS][k+1][j];
										prob = num / sen_prob;
										update_gen_ecounts(htag, hstag, RIGHT, ONE_KID, atag, astag, prob);

										num = alpha[hidx][hstag][R][TWO_OR_MORE][i][j] *
													 get_gen_prob(htag, hstag, RIGHT, TWO_OR_MORE, atag, astag) *
													 get_constr_value(hidx, aidx, TWO_OR_MORE, s) *
													 beta[hidx][hstag][R][TWO_OR_MORE][i][k] *
													 beta[aidx][astag][F][NO_KIDS][k+1][j];
										prob = num / sen_prob;
										update_gen_ecounts(htag, hstag, RIGHT, TWO_OR_MORE, atag, astag, prob);
								}
							}
						}


						hidx = j;
						htag = s.words[hidx].tag;
						for(int l = i; l <= k; l++)
						{
							int aidx = l;
							int atag = s.words[aidx].tag;
							for(int hstag = 0; hstag < T; hstag++)
							{
								for(int astag = 0; astag < T; astag++)
								{
										num = alpha[hidx][hstag][L][ONE_KID][i][j] *
											  get_gen_prob(htag, hstag, LEFT, NO_KIDS, atag, astag) *
											  get_constr_value(hidx, aidx, NO_KIDS, s) *
											  beta[hidx][hstag][L][NO_KIDS][k+1][j] *
											  beta[aidx][astag][F][NO_KIDS][i][k];
										prob = num / sen_prob;
										update_gen_ecounts(htag, hstag, LEFT, NO_KIDS, atag, astag, prob);

										num = alpha[hidx][hstag][L][TWO_OR_MORE][i][j] *
											  get_gen_prob(htag, hstag, LEFT, ONE_KID, atag, astag) *
											  get_constr_value(hidx, aidx, ONE_KID, s) *
											  beta[hidx][hstag][L][ONE_KID][k+1][j] *
											  beta[aidx][astag][F][NO_KIDS][i][k];
										prob = num / sen_prob;
										update_gen_ecounts(htag, hstag, LEFT, ONE_KID, atag, astag, prob);

										num = alpha[hidx][hstag][L][TWO_OR_MORE][i][j] *
											  get_gen_prob(htag, hstag, LEFT, TWO_OR_MORE, atag, astag) *
											  get_constr_value(hidx, aidx, TWO_OR_MORE, s) *
											  beta[hidx][hstag][L][TWO_OR_MORE][k+1][j] *
											  beta[aidx][astag][F][NO_KIDS][i][k];
										prob = num / sen_prob;
										update_gen_ecounts(htag, hstag, LEFT, TWO_OR_MORE, atag, astag, prob);
								}
							}
						}
					}
					
					//--------------------------------------------------------------------------------
					for(int hidx = i; hidx <= j; hidx++)
					{
						int htag = s.words[hidx].tag;
						int word = s.words[hidx].word;
						for(int hstag = 0; hstag < T; hstag++)
							for(int lkids = NO_KIDS; lkids <=TWO_OR_MORE; lkids++)
								for(int rkids = NO_KIDS; rkids <=TWO_OR_MORE; rkids++)
								{
									num = alpha[hidx][hstag][F][NO_KIDS][i][j] *
										stop_prob[htag][hstag][RIGHT][rkids][STOP] *
										beta[hidx][hstag][R][rkids][hidx][j] *
										stop_prob[htag][hstag][LEFT][lkids][STOP] *
										beta[hidx][hstag][L][lkids][i][hidx] /
										emissions[htag][hstag][word];
									prob = num / sen_prob;
									stop_LHS[htag][hstag][RIGHT][rkids] += prob;
									stop_RHS[htag][hstag][RIGHT][rkids][STOP] += prob;
									stop_LHS[htag][hstag][LEFT][lkids] += prob;
									stop_RHS[htag][hstag][LEFT][lkids][STOP] += prob;
								}
					}
				}
	}
}

void model::var_bound()
{
	cout<<"partition function: "<<partition<<endl;
	double bound = 0;
	bound += partition;

	double KL = 0;
	for(int htag = 0; htag < tsize; htag++)
	{
		for(int hstag = 0; hstag < T; hstag++)
		{
			for(int dir = LEFT; dir <= RIGHT; dir++)
			{
				for(int kids = NO_KIDS; kids <= TWO_OR_MORE; kids++)
				{
					KL = 0;
					KL += (gammaln(stp_hyp * 2 + stop_LHS[htag][hstag][dir][kids]) 
						  - gammaln(stp_hyp * 2) );
					for(int stype = STOP; stype <= CONTINUE; stype++)
					{
						KL -= ( gammaln(stp_hyp + stop_RHS[htag][hstag][dir][kids][stype]) - gammaln(stp_hyp) );
						KL += ( stop_RHS[htag][hstag][dir][kids][stype] * 
							    (digamma(stp_hyp + stop_RHS[htag][hstag][dir][kids][stype]) - 
								 digamma(stp_hyp * 2 + stop_LHS[htag][hstag][dir][kids])) );
					}
					bound -= KL;

					KL = 0;
					KL += ( gammaln(tag_hyp * tsize + choose_tag_LHS[htag][hstag][dir][kids]) 
						  - gammaln(tag_hyp * tsize) );
					for(int atag = 0; atag < tsize; atag++)
					{
						KL -= ( gammaln(tag_hyp + choose_tag_RHS[htag][hstag][dir][kids][atag]) - gammaln(tag_hyp) );
						KL += ( choose_tag_RHS[htag][hstag][dir][kids][atag] * 
							    (digamma(tag_hyp + choose_tag_RHS[htag][hstag][dir][kids][atag]) - 
								 digamma(tag_hyp * tsize + choose_tag_LHS[htag][hstag][dir][kids])) );
					}
					bound -= KL;

					for(int atag = 0; atag < tsize; atag++)
					{
						KL = 0;
						KL += ( gammaln(alpha_1 + choose_stag_LHS[htag][hstag][dir][kids][atag]) 
							  - gammaln(alpha_1) );
						for(int astag = 0; astag < T; astag++)
						{
							KL -= ( gammaln(alpha_1 * betas[atag][astag] + choose_stag_RHS[htag][hstag][dir][kids][atag][astag]) 
								  - gammaln(alpha_1 * betas[atag][astag]) );
							KL += ( choose_stag_RHS[htag][hstag][dir][kids][atag][astag] * 
									(digamma(alpha_1 * betas[atag][astag] + choose_stag_RHS[htag][hstag][dir][kids][atag][astag]) - 
									 digamma(alpha_1 + choose_stag_LHS[htag][hstag][dir][kids][atag])) );
						}
						bound -= KL;
					}
				}
			}
			KL = 0;
			KL += ( gammaln(ems_hyp * wsize + tags[htag][hstag]) - gammaln(ems_hyp * wsize) );
			for(int w = 0; w < wsize; w++)
			{
				KL -= ( gammaln(ems_hyp + tag_words[htag][hstag][w]) - gammaln(ems_hyp) );
				KL += ( tag_words[htag][hstag][w] * 
					    (digamma(ems_hyp + tag_words[htag][hstag][w]) - 
						 digamma(ems_hyp * wsize + tags[htag][hstag])) );
			}
			bound -= KL;
		}
	}

	for(int tag = 0; tag < tsize; tag++)
	{
		double rem_stick = 1;
		double prob = 1;
		for(int stag = 0; stag < (T-1); stag++)
		{
			double x = betas[tag][stag] / rem_stick;
			prob *= ( ( gamma(1+alpha_0) * pow(1-x,alpha_0-1) ) / gamma(alpha_0) ) * (1 / rem_stick);
			rem_stick -= betas[tag][stag];
		}
		bound += log(prob);
	}
	cout<<"\nvariational bound: "<<bound<<endl;
}


void model::M_Step()
{
	for(int htag = 0; htag < tsize; htag++)
	{
		for(int hstag = 0; hstag < T; hstag++)
		{
			for(int dir = LEFT; dir <= RIGHT; dir++)
			{
				for(int kids = NO_KIDS; kids <= TWO_OR_MORE; kids++)
				{
					for(int stype = STOP; stype <= CONTINUE; stype++)
					{
						stop_prob[htag][hstag][dir][kids][stype] 
						= stop_LHS[htag][hstag][dir][kids] == 0 ? 0 : 
							stop_RHS[htag][hstag][dir][kids][stype]/stop_LHS[htag][hstag][dir][kids];
					}

					for(int atag = 0; atag < tsize; atag++)
					{
						choose_tag_prob[htag][hstag][dir][kids][atag] 
						= choose_tag_LHS[htag][hstag][dir][kids] == 0 ? 0 :
							choose_tag_RHS[htag][hstag][dir][kids][atag] / choose_tag_LHS[htag][hstag][dir][kids];
						for(int astag = 0; astag < T; astag++)
							choose_stag_prob[htag][hstag][dir][kids][atag][astag] 
						= choose_stag_LHS[htag][hstag][dir][kids][atag] == 0 ? 0 :
							choose_stag_RHS[htag][hstag][dir][kids][atag][astag] / 
							choose_stag_LHS[htag][hstag][dir][kids][atag];
					}
				}
			}
		
			for(int w = 0; w < wsize; w++)
				emissions[htag][hstag][w] = tags[htag][hstag] == 0 ? 0 : tag_words[htag][hstag][w] / tags[htag][hstag];
		}
	}
}

void model::var_M_Step()
{
	for(int htag = 0; htag < tsize; htag++)
	{
		for(int hstag = 0; hstag < T; hstag++)
		{
			for(int dir = LEFT; dir <= RIGHT; dir++)
			{
				for(int kids = NO_KIDS; kids <= TWO_OR_MORE; kids++)
				{
					for(int stype = STOP; stype <= CONTINUE; stype++)
					{
						stop_prob[htag][hstag][dir][kids][stype] 
					    = exp(digamma(stp_hyp + stop_RHS[htag][hstag][dir][kids][stype]) 
							  - digamma(stp_hyp * 2 + stop_LHS[htag][hstag][dir][kids]));
					}

					for(int atag = 0; atag < tsize; atag++)
					{
						choose_tag_prob[htag][hstag][dir][kids][atag] 
						= exp(digamma(tag_hyp + choose_tag_RHS[htag][hstag][dir][kids][atag]) 
							  - digamma(tag_hyp * tsize + choose_tag_LHS[htag][hstag][dir][kids]));
						for(int astag = 0; astag < T; astag++)
						{
							choose_stag_prob[htag][hstag][dir][kids][atag][astag] 
							= (betas[atag][astag] + choose_stag_RHS[htag][hstag][dir][kids][atag][astag]) == 0 ? 0 : 
								exp(digamma(alpha_1 * betas[atag][astag] + choose_stag_RHS[htag][hstag][dir][kids][atag][astag]) 
								  - digamma(alpha_1 + choose_stag_LHS[htag][hstag][dir][kids][atag]));
						}
					}
				}
			}
		
			for(int w = 0; w < wsize; w++)
			{
				emissions[htag][hstag][w] = exp(digamma(ems_hyp + tag_words[htag][hstag][w]) - digamma(ems_hyp * wsize + tags[htag][hstag]));
			}
		}
	}
	//log.close();
	//cout<<stotal<<endl;
}

double model::compute_beta_objective(int tag)
{
	double objective = 0;

	double rem_stick = 1;
	double prob = 1;
	for(int stag = 0; stag < (T-1); stag++)
	{
		double x = betas[tag][stag] / rem_stick;
		prob *= ( ( gamma(1+alpha_0) * pow(1-x,alpha_0-1) ) / gamma(alpha_0) ) * (1 / rem_stick);
		rem_stick -= betas[tag][stag];
	}
	objective += log(prob);


	double sum = 0;
	for(int htag = 0; htag < tsize; htag++)
	{
		for(int hstag = 0; hstag < T; hstag++)
		{
			for(int dir = LEFT; dir <= RIGHT; dir++)
			{
				for(int kids = NO_KIDS; kids <= TWO_OR_MORE; kids++)
				{
					for(int stag = 0; stag < T; stag++)
					{
						sum -= log(gamma(alpha_1 * betas[tag][stag]));
						sum += (alpha_1 * betas[tag][stag] - 1) *
							   ( digamma(alpha_1 * betas[tag][stag] + choose_stag_RHS[htag][hstag][dir][kids][tag][stag]) 
					             - digamma(alpha_1 + choose_stag_LHS[htag][hstag][dir][kids][tag]) );
					}
				}
			}
		}
	}
	objective += sum;

	return objective;
}

double model::compute_beta_gradient(int tag, int stag)
{
	double gradient = 0;

	double rem_stick = 1;
	for(int i = 0; i <= stag; i++)
		rem_stick -= betas[tag][i];

	for(int i = stag + 1; i < (T - 1); i++)
	{
		gradient += 1 / rem_stick;
		rem_stick -= betas[tag][i];
	}

	gradient -= (alpha_0 - 1) / rem_stick;

	double sum = 0;
	for(int htag = 0; htag < tsize; htag++)
	{
		for(int hstag = 0; hstag < T; hstag++)
		{
			for(int dir = LEFT; dir <= RIGHT; dir++)
			{
				for(int kids = NO_KIDS; kids <= TWO_OR_MORE; kids++)
				{
					sum +=( digamma(alpha_1 * betas[tag][stag] + choose_stag_RHS[htag][hstag][dir][kids][tag][stag]) 
					- digamma(alpha_1 + choose_stag_LHS[htag][hstag][dir][kids][tag])
					- digamma(alpha_1 * betas[tag][stag]) ) 
					- ( digamma(alpha_1 * betas[tag][T-1] + choose_stag_RHS[htag][hstag][dir][kids][tag][T-1]) 
					- digamma(alpha_1 + choose_stag_LHS[htag][hstag][dir][kids][tag])
					- digamma(alpha_1 * betas[tag][T-1]) );
				}
			}
		}
	}
	gradient += alpha_1 * sum;

	return gradient;
}

void model::estimate_betas()
{
	//initialize
	for(int t = 0; t < tsize; t++)
		for(int st = 0; st < T; st++)
			betas[t][st] = 1.0 / T;

	//gradient serach
	for(int tag = 0; tag < tsize; tag++)
	{
		vector<double> newbetas(T);
		vector<double> grad_arr(T);
		double mu = 0.0001; //step size
		double gradient;
		double last_gradient=99999999;
		double epsilon = 0.00001;
		bool changed;
		double sum_beta;
		int iter = 0;
		do
		{
			gradient = 0;
			sum_beta = 0;

			int stag;
			for(stag = 0; stag < (T - 1); stag++)
			{
				double grad = compute_beta_gradient(tag, stag);
				if( (grad_arr[stag]>0 && grad<0) || (grad>0 && grad_arr[stag]<0) )
				{
					break;
					//newbetas[stag] = betas[tag][stag];
				}
				else
				{
					newbetas[stag] = (betas[tag][stag] + mu * grad) > epsilon ? (betas[tag][stag] + mu * grad): epsilon;
					gradient += grad * grad;
					grad_arr[stag] = grad;
				}
				sum_beta += newbetas[stag];
			}
			if(stag < (T - 1))
				break;
			gradient = sqrt(gradient);

			//project to probability simplex
			if(sum_beta > (1-epsilon) )
			{
				int count = T - 1;
				while(sum_beta > 1)
				{
					double subtract = (sum_beta - (1-epsilon)) / count;
					sum_beta = 0;
					count = 0;
					for(int t = 0; t < (T - 1); t++)
					{
						newbetas[t] = ((newbetas[t] - subtract) >= epsilon ? (newbetas[t] - subtract) : epsilon );
						sum_beta+= newbetas[t];
						if(newbetas[t] > epsilon) count++;
					}
				}
			}
			//final beta
			newbetas[T - 1] = 1 - sum_beta;

			vector<double> direction(T);
			vector<double> oldbetas(T);
			for( int i =0; i < T; i++)
			{
				oldbetas[i]  = betas[tag][i];
				direction[i] = newbetas[i] - oldbetas[i];
			}

			double old_obj = compute_beta_objective(tag);

			for(int stag = 0; stag < (T); stag++)
				betas[tag][stag] = newbetas[stag];

			double new_obj = compute_beta_objective(tag);

			if(new_obj < old_obj)
			{
				//step size reduction
				double factor = 0.5;
				double threshold = 0.001;
				double step = 0.2;
				int it;
				for(it = 0; it < 50; it++)
				{
					for(int i = 0; i < T; i++)
						betas[tag][i] = oldbetas[i] + step * direction[i];
					new_obj = compute_beta_objective(tag);
					if(new_obj > old_obj)
						break;
					step *= factor;
				}
				if(it == 50)
					for(int i = 0; i < T; i++)
						betas[tag][i] = oldbetas[i];

			}

			changed = false;
			for(int stag = 0; stag < (T-1); stag++)
				if(betas[tag][stag] != oldbetas[stag])
					changed = true;

			iter ++;
			last_gradient = gradient;
		}
		while(gradient > 0.1 && changed && iter <= 40);
	}
}

void model::print_betas()
{
	for(int tag = 0; tag < tsize; tag++)
	{
		cout<<d.gtags[tag]<<" : ";
		for(int stag = 0; stag < T; stag++)
		{
			cout<<betas[tag][stag]<<"\t";
		}
		cout<<endl;
	}
}

void model::gradient_E_Step()
{
	//initialize
	for(int ctype = 0; ctype < constr_count; ctype++)
		constr_ecount[ctype] = 0;
	double etotal = 0;
	for(int sidx = 0; sidx < d.sents.size(); sidx++)
	{
		double old = etotal;
		sentence s = d.sents[sidx];
		int senlen = s.words.size();

		Inside(s);
		Outside(s);

		double sen_prob = beta[senlen-1][0][F][NO_KIDS][0][senlen-1];
		if(sen_prob <= 0)
			cout<<"\tsentence has no prob. number = "<<sidx+1<<"\n";
		double prob, num;
		partition += log(sen_prob);

		for(int size = 1; size <= senlen; size++)
			for(int i = 0; i <= senlen - size ; i++)
				{
					int j = i + size - 1;
					for(int k = i; k < j; k++)
					{
						int hidx = i;
						int htag = s.words[hidx].tag;
						for(int r = k+1; r <=j; r++)
						{
							int aidx = r;
							int atag = s.words[aidx].tag;
							for(int hstag = 0; hstag < T; hstag++)
							{
								for(int astag = 0; astag < T; astag++)
								{
										num = alpha[hidx][hstag][R][ONE_KID][i][j] *
													 get_gen_prob(htag, hstag, RIGHT, NO_KIDS, atag, astag) *
													 get_constr_value(hidx, aidx, NO_KIDS, s) *
													 beta[hidx][hstag][R][NO_KIDS][i][k] *
													 beta[aidx][astag][F][NO_KIDS][k+1][j];
										prob = num / sen_prob;
										for(int ctype = 0; ctype < constr_count; ctype++)
											constr_ecount[ctype] += prob * (this->*constr_func[ctype])(hidx, aidx, NO_KIDS, s);
										etotal += prob;

										num = alpha[hidx][hstag][R][TWO_OR_MORE][i][j] *
													 get_gen_prob(htag, hstag, RIGHT, ONE_KID, atag, astag) *
													 get_constr_value(hidx, aidx, ONE_KID, s) *
													 beta[hidx][hstag][R][ONE_KID][i][k] *
													 beta[aidx][astag][F][NO_KIDS][k+1][j];
										prob = num / sen_prob;
										for(int ctype = 0; ctype < constr_count; ctype++)
											constr_ecount[ctype] += prob * (this->*constr_func[ctype])(hidx, aidx, ONE_KID, s);
										etotal += prob;

										num = alpha[hidx][hstag][R][TWO_OR_MORE][i][j] *
													 get_gen_prob(htag, hstag, RIGHT, TWO_OR_MORE, atag, astag) *
													 get_constr_value(hidx, aidx, TWO_OR_MORE, s) *
													 beta[hidx][hstag][R][TWO_OR_MORE][i][k] *
													 beta[aidx][astag][F][NO_KIDS][k+1][j];
										prob = num / sen_prob;
										for(int ctype = 0; ctype < constr_count; ctype++)
											constr_ecount[ctype] += prob * (this->*constr_func[ctype])(hidx, aidx, TWO_OR_MORE, s);
										etotal += prob;
									
								}
							}
						}
								
						hidx = j;
						htag = s.words[hidx].tag;
						for(int l = i; l <= k; l++)
						{
							int aidx = l;
							int atag = s.words[aidx].tag;
							for(int hstag = 0; hstag < T; hstag++)
							{
								for(int astag = 0; astag < T; astag++)
								{
										num = alpha[hidx][hstag][L][ONE_KID][i][j] *
											  get_gen_prob(htag, hstag, LEFT, NO_KIDS, atag, astag) *
											  get_constr_value(hidx, aidx, NO_KIDS, s) *
											  beta[hidx][hstag][L][NO_KIDS][k+1][j] *
											  beta[aidx][astag][F][NO_KIDS][i][k];
										prob = num / sen_prob;
										for(int ctype = 0; ctype < constr_count; ctype++)
											constr_ecount[ctype] += prob * (this->*constr_func[ctype])(hidx, aidx, NO_KIDS, s);
										etotal += prob;

										num = alpha[hidx][hstag][L][TWO_OR_MORE][i][j] *
											  get_gen_prob(htag, hstag, LEFT, ONE_KID, atag, astag) *
											  get_constr_value(hidx, aidx, ONE_KID, s) *
											  beta[hidx][hstag][L][ONE_KID][k+1][j] *
											  beta[aidx][astag][F][NO_KIDS][i][k];
										prob = num / sen_prob;
										for(int ctype = 0; ctype < constr_count; ctype++)
											constr_ecount[ctype] += prob * (this->*constr_func[ctype])(hidx, aidx, ONE_KID, s);
										etotal += prob;

										num = alpha[hidx][hstag][L][TWO_OR_MORE][i][j] *
											  get_gen_prob(htag, hstag, LEFT, TWO_OR_MORE, atag, astag) *
											  get_constr_value(hidx, aidx, TWO_OR_MORE, s) *
											  beta[hidx][hstag][L][TWO_OR_MORE][k+1][j] *
											  beta[aidx][astag][F][NO_KIDS][i][k];
										prob = num / sen_prob;
										for(int ctype = 0; ctype < constr_count; ctype++)
											constr_ecount[ctype] += prob * (this->*constr_func[ctype])(hidx, aidx, TWO_OR_MORE, s);
										etotal += prob;
									
								}
							}
						}
					}
					
				}
	}
	for(int ctype = 0; ctype < constr_count; ctype++)
		cout<<constr_bound[ctype]<<" "<<constr_ecount[ctype]<<" "<<etotal<<endl;
}

void model::compute_gradient()
{
	gradient_E_Step();
	for(int ctype = 0; ctype < constr_count; ctype++)
		gradient[ctype] = constr_bound[ctype] - constr_ecount[ctype];
}

void model::gradient_search()
{
	double gamma = 0.0001;
	
	for(int i = 0; i < constr_count; i++)
		lembda[i] = 0.0;
	compute_gradient();

	int iter = 1;
	while(iter <= 5)
	{
		cout<<"gradient iteration "<<iter<<endl;
		for(int i = 0; i < constr_count; i++)
		{
			cout<<lembda[i]<<"\t"<<gamma * gradient[i]<<endl;
			lembda[i]   = (lembda[i] - gamma * gradient[i]) > 0 ? (lembda[i] - gamma * gradient[i]) : 0;
		}
		compute_gradient();

		iter ++;
	}
	


}
void model::CKY(sentence &s)
{
	s.pdeps.clear();
	int senlen = s.words.size();
	
	//initialize
	for(int n = 0; n < senlen; n++)
		for(int stype = R; stype <= F; stype++)
			for(int i = 0; i < senlen; i++)
				for(int j = 0; j < senlen; j++)
					for(int stag = 0; stag < T; stag++)
						for(int kids = NO_KIDS; kids <= TWO_OR_MORE; kids++)
						{
							back[n][stag][stype][kids][i][j].prob    =  0;
							back[n][stag][stype][kids][i][j].argstag = -1;
							back[n][stag][stype][kids][i][j].argidx  = -1;
							back[n][stag][stype][kids][i][j].split   = -1;
							back[n][stag][stype][kids][i][j].kids    = -1;
						}

	//base case
	for(int n = 0; n < senlen; n++)
	{
		int word = s.words[n].word;
		int tag  = s.words[n].tag;
		for(int stag = 0; stag < T; stag++)
		{
			back[n][stag][R][NO_KIDS][n][n].prob = emissions[tag][stag][word];
		}
	}

	//recurse
	double prob;
	for(int size = 1; size < senlen; size++)
		for(int i = 0; i < senlen - size ; i++)
			{
				int j = i + size - 1;
				for(int k = i; k < j; k++)
				{
					for(int l = i; l <= k; l++)
					{
						for(int r = k+1; r <=j; r++)
						{
							int hidx = l;
							int htag = s.words[hidx].tag;
							for(int hstag = 0; hstag < T; hstag++)
							{
								int aidx = r;	
								int atag = s.words[aidx].tag;
								for(int astag = 0; astag < T; astag++)
								{
									prob = get_gen_prob(htag, hstag, RIGHT, NO_KIDS, atag, astag) *
									   get_constr_value(hidx, aidx, NO_KIDS, s) *
									   back[hidx][hstag][R][NO_KIDS][i][k].prob *
									   back[aidx][astag][F][NO_KIDS][k+1][j].prob;
									if(prob > back[hidx][hstag][R][ONE_KID][i][j].prob)
									{
										back[hidx][hstag][R][ONE_KID][i][j].prob    = prob;
										back[hidx][hstag][R][ONE_KID][i][j].argidx  = aidx;
										back[hidx][hstag][R][ONE_KID][i][j].argstag = astag;
										back[hidx][hstag][R][ONE_KID][i][j].split   = k;
										back[hidx][hstag][R][ONE_KID][i][j].kids    = NO_KIDS;
									}

									prob = get_gen_prob(htag, hstag, RIGHT, ONE_KID, atag, astag) *
									   get_constr_value(hidx, aidx, ONE_KID, s) *
									   back[hidx][hstag][R][ONE_KID][i][k].prob *
									   back[aidx][astag][F][NO_KIDS][k+1][j].prob;
									if(prob > back[hidx][hstag][R][TWO_OR_MORE][i][j].prob)
									{
										back[hidx][hstag][R][TWO_OR_MORE][i][j].prob = prob;
										back[hidx][hstag][R][TWO_OR_MORE][i][j].argidx = aidx;
										back[hidx][hstag][R][TWO_OR_MORE][i][j].argstag = astag;
										back[hidx][hstag][R][TWO_OR_MORE][i][j].split = k;
										back[hidx][hstag][R][TWO_OR_MORE][i][j].kids = ONE_KID;
									}	

									prob = get_gen_prob(htag, hstag, RIGHT, TWO_OR_MORE, atag, astag) *
									   get_constr_value(hidx, aidx, TWO_OR_MORE, s) *
									   back[hidx][hstag][R][TWO_OR_MORE][i][k].prob *
									   back[aidx][astag][F][NO_KIDS][k+1][j].prob;
									if(prob > back[hidx][hstag][R][TWO_OR_MORE][i][j].prob)
									{
										back[hidx][hstag][R][TWO_OR_MORE][i][j].prob = prob;
										back[hidx][hstag][R][TWO_OR_MORE][i][j].argidx = aidx;
										back[hidx][hstag][R][TWO_OR_MORE][i][j].argstag = astag;
										back[hidx][hstag][R][TWO_OR_MORE][i][j].split = k;
										back[hidx][hstag][R][TWO_OR_MORE][i][j].kids = TWO_OR_MORE;
									}
									//assert(beta[hidx][head][HALFL][i][j] >= 0);
								}
							}
							
							hidx = r;
							htag = s.words[hidx].tag;
							for(int hstag = 0; hstag < T; hstag++)
							{
								int aidx = l;	
								int atag = s.words[aidx].tag;
								for(int astag = 0; astag < T; astag++)
								{		
									prob = get_gen_prob(htag, hstag, LEFT, NO_KIDS, atag, astag) *
									   get_constr_value(hidx, aidx, NO_KIDS, s) *
									   back[hidx][hstag][L][NO_KIDS][k+1][j].prob *
									   back[aidx][astag][F][NO_KIDS][i][k].prob;
									if( prob > back[hidx][hstag][L][ONE_KID][i][j].prob )
									{
										back[hidx][hstag][L][ONE_KID][i][j].prob = prob;
										back[hidx][hstag][L][ONE_KID][i][j].argstag = astag;
										back[hidx][hstag][L][ONE_KID][i][j].argidx = aidx;
										back[hidx][hstag][L][ONE_KID][i][j].split = k;
										back[hidx][hstag][L][ONE_KID][i][j].kids = NO_KIDS;
									}

									prob = get_gen_prob(htag, hstag, LEFT, ONE_KID, atag, astag) *
									   get_constr_value(hidx, aidx, ONE_KID, s) *
									   back[hidx][hstag][L][ONE_KID][k+1][j].prob *
									   back[aidx][astag][F][NO_KIDS][i][k].prob;
									if( prob > back[hidx][hstag][L][TWO_OR_MORE][i][j].prob )
									{
										back[hidx][hstag][L][TWO_OR_MORE][i][j].prob = prob;
										back[hidx][hstag][L][TWO_OR_MORE][i][j].argidx = aidx;
										back[hidx][hstag][L][TWO_OR_MORE][i][j].argstag = astag;
										back[hidx][hstag][L][TWO_OR_MORE][i][j].split = k;
										back[hidx][hstag][L][TWO_OR_MORE][i][j].kids = ONE_KID;
									}
									prob = get_gen_prob(htag, hstag, LEFT, TWO_OR_MORE, atag, astag) *
									   get_constr_value(hidx, aidx, TWO_OR_MORE, s) *
									   back[hidx][hstag][L][TWO_OR_MORE][k+1][j].prob *
									   back[aidx][astag][F][NO_KIDS][i][k].prob;
									if( prob > back[hidx][hstag][L][TWO_OR_MORE][i][j].prob )
									{
										back[hidx][hstag][L][TWO_OR_MORE][i][j].prob = prob;
										back[hidx][hstag][L][TWO_OR_MORE][i][j].argidx = aidx;
										back[hidx][hstag][L][TWO_OR_MORE][i][j].argstag = astag;
										back[hidx][hstag][L][TWO_OR_MORE][i][j].split = k;
										back[hidx][hstag][L][TWO_OR_MORE][i][j].kids = TWO_OR_MORE;
									}
								}
							}
						}
					}
				}
				
				//--------------------------------------------------------------------------------
				for(int hidx = i; hidx <= j; hidx++)
				{
					int htag = s.words[hidx].tag;
					for(int hstag = 0; hstag < T; hstag++)
					{
						for(int kids = NO_KIDS; kids <= TWO_OR_MORE; kids++)
						{
							prob = stop_prob[htag][hstag][RIGHT][kids][STOP] *
								back[hidx][hstag][R][kids][i][j].prob;
							if(prob > back[hidx][hstag][L][NO_KIDS][i][j].prob)
							{
								back[hidx][hstag][L][NO_KIDS][i][j].prob = prob;
								back[hidx][hstag][L][NO_KIDS][i][j].kids = kids;
							}
						}

						for(int kids = NO_KIDS; kids <= TWO_OR_MORE; kids++)
						{
							prob = stop_prob[htag][hstag][LEFT][kids][STOP] *
								back[hidx][hstag][L][kids][i][j].prob;
							if(prob > back[hidx][hstag][F][NO_KIDS][i][j].prob)
							{
								back[hidx][hstag][F][NO_KIDS][i][j].prob = prob;
								back[hidx][hstag][F][NO_KIDS][i][j].kids = kids;
							}
						}
					}
				}
			}

	//Dealing with the root symbol
	int end = senlen - 1;
	int htag = s.words[end].tag;
	int hstag = 0 ; //root tag has only one subtag
	back[end][hstag][L][NO_KIDS][end][end].prob = 1.0;//stop_prob[htag][hstag][RIGHT][NO_KIDS][STOP];
	int hidx = end;
	int i = 0; 
	int j = end;
	int k = j - 1;
	for(int aidx = 0; aidx < end; aidx++)
	{
		int atag = s.words[aidx].tag;
		for(int astag = 0; astag < T; astag++)
		{
			prob = get_gen_prob(htag, hstag, LEFT, NO_KIDS, atag, astag) *
			   get_constr_value(hidx, aidx, NO_KIDS, s) *
			   back[hidx][hstag][L][NO_KIDS][k+1][j].prob *
			   back[aidx][astag][F][NO_KIDS][i][k].prob;
			if(prob > back[hidx][hstag][L][ONE_KID][i][j].prob)
			{
				back[hidx][hstag][L][ONE_KID][i][j].prob = prob;
				back[hidx][hstag][L][ONE_KID][i][j].argidx = aidx;
				back[hidx][hstag][L][ONE_KID][i][j].argstag = astag;
				back[hidx][hstag][L][ONE_KID][i][j].split = k;
				back[hidx][hstag][L][ONE_KID][i][j].kids = NO_KIDS;
			}
		}
	}

	prob = stop_prob[htag][hstag][LEFT][ONE_KID][STOP] *
		back[hidx][hstag][L][ONE_KID][i][j].prob;
	back[hidx][hstag][F][NO_KIDS][i][j].prob = prob;
	back[hidx][hstag][F][NO_KIDS][i][j].kids = ONE_KID; 

	//populate dependecies
	get_pdeps(senlen - 1, hstag, F, NO_KIDS, 0, senlen - 1, s.pdeps, s.words);

}

void model::get_pdeps(int hidx, int hstag, Seal type, Kids kids, int i, int j, vector<dep> &deps, vector<token> &words)
{
	if(i>=j)
	{
		words[i].subtag = hstag;
		return;
	}

	int astag   = back[hidx][hstag][type][kids][i][j].argstag;
	int split   = back[hidx][hstag][type][kids][i][j].split;
	Kids bkids  = (Kids)(back[hidx][hstag][type][kids][i][j].kids);		

	switch(type)
	{
		case R:
			{
				dep d;
				d.head_idx = hidx;
				d.arg_idx  = back[hidx][hstag][type][kids][i][j].argidx;
				if (split >= 0)
				{
					get_pdeps(hidx, hstag, type, bkids,  i, split, deps, words);
					get_pdeps(d.arg_idx, astag, F, NO_KIDS, split+1, j, deps, words);
				}
				deps.push_back(d);
			}
			break;
		
		case L:
			{
				if (back[hidx][hstag][type][kids][i][j].split >= 0)
				{
					dep d;
					d.head_idx = hidx;
					d.arg_idx  = back[hidx][hstag][type][kids][i][j].argidx;
					if (split >= 0)
					{
						get_pdeps(hidx, hstag, type, bkids, split+1, j, deps, words);
						get_pdeps(d.arg_idx, astag, F, NO_KIDS, i, split, deps, words);
					}
					deps.push_back(d);
				}
				else
				{
					get_pdeps(hidx, hstag, R, bkids, i, j, deps, words);
				}
			}
			break;

		case F: 
			{
				get_pdeps(hidx, hstag, L, bkids, i, j, deps, words);
			}
			break;
		
		default: return;
	}
	
}

void model::Annotate()
{
  for(int i = 0; i < d.sents.size(); i++)
	  CKY(d.sents[i]);
}


void model::learn()
{
	for(int round = 1; round <= 50; round++)
	{
		if ( constr_count > 0 && round % 1 == 0)
		{
			cout<<"gradient search .....\n";
			gradient_search();
		}
		
		if(round % 1 == 0)
		{
			Annotate();
			output();
		}
		
		cout<<"E_Step\n";
        E_Step();
        cout<<"M_Step\n";
        var_M_Step();

		//for(int ctype = 0; ctype < constr_count; ctype++)
		//	lembda[ctype] = 0.0;
		//if(round % 1 == 0)
		//{
		//	Annotate();
		//	output();
		//}
		
		if(strcmp(d.config["beta"].c_str(), "true") == 0)
        {
			var_bound();
			cout<<"estimate betas\n";
            estimate_betas();
            print_betas();
			var_bound();
        }
        else
        {
            var_bound();
        }
		cout<<"round .... "<<round<<endl;
	}
}

void model::output()
{
	d.print_results(true);
	d.print_results(false);
	d.write_output();
	d.write_stats(true);
	d.write_stats(false);
	d.print_subtags();
	print_subtags();
}
int main()
{
	model m;
	cout<<"Learning ..... \n";
	m.learn();
	cout<<"Done ..... \n";	
	return 0;
}


